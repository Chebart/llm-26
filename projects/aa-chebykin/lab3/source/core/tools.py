import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import Context, FastMCP
from pglast import ast, parse_sql
from pglast.visitors import Visitor
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import BaseModel, Field

from core.config import Settings
from core.retrieval import Retriever
from core.tracing import Tracer


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)


class PageArgs(BaseModel):
    document_id: str = Field(pattern=r"^pg17-p\d{4}$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=6000, ge=100, le=12000)


class SchemaArgs(BaseModel):
    schema_name: str = Field(default="public", pattern="^public$", alias="schema")


class SqlArgs(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


ARGUMENTS = {
    "search_docs": SearchArgs,
    "read_doc_page": PageArgs,
    "inspect_schema": SchemaArgs,
    "execute_sql": SqlArgs,
    "explain_sql": SqlArgs,
}
MCP_TOOLS = {"search_docs", "read_doc_page", "inspect_schema"}


class ToolResult(BaseModel):
    call_id: str
    tool_name: str
    status: str
    data: Any = None
    error: dict | None = None
    artifact_id: str | None = None
    truncated: bool = False
    duration_ms: float = 0


class SchemaGuard(Visitor):
    def visit_RangeVar(self, ancestors: Any, node: ast.RangeVar) -> None:
        if node.catalogname or node.schemaname:
            raise ValueError("Use unqualified table names inside the isolated exercise schema")


def validate_sql(query: str, explain: bool = False) -> str:
    statements = parse_sql(query)

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed")

    node = statements[0].stmt
    kind = type(node).__name__
    allowed = (
        {"SelectStmt"}
        if explain
        else {
            "SelectStmt",
            "InsertStmt",
            "UpdateStmt",
            "DeleteStmt",
            "CreateStmt",
            "AlterTableStmt",
        }
    )

    if kind not in allowed:
        raise ValueError(f"Statement {kind} is not allowed in the exercise database")

    SchemaGuard()(node)

    return kind


def session_schema(session_id: str) -> str:
    return "task_" + hashlib.sha256(session_id.encode()).hexdigest()[:20]


class ToolService:
    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        self.settings = settings
        self.retriever = Retriever(settings)
        self.tracer = tracer or Tracer(settings.sessions / "tools")
        self.events: dict[str, list[dict]] = {}
        self.turns: dict[str, dict] = {}

    def schemas(self) -> list[dict]:
        descriptions = self.settings.prompts()["tools"]

        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": descriptions[name],
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in ARGUMENTS.items()
        ]

    def reset(self, session_id: str) -> str:
        if not self.settings.admin_url:
            raise ValueError(
                "Set LAB3_DATABASE_ADMIN_URL in .env and start the exercise PostgreSQL"
            )

        schema = session_schema(session_id)

        # Reset применяется только к вычисленной схеме этой задачи, никогда ко всей БД.
        with psycopg.connect(self.settings.admin_url, autocommit=True) as connection:
            if not connection.execute(
                "SELECT to_regprocedure('public.seed_exercise()')"
            ).fetchone()[0]:
                raise ValueError(
                    "Apply resources/database.sql with psql as database admin before reset"
                )

            password = self._session_password(session_id)
            exists = connection.execute(
                "SELECT 1 FROM pg_roles WHERE rolname=%s", (schema,)
            ).fetchone()
            connection.execute(
                sql.SQL(
                    "ALTER ROLE {} LOGIN PASSWORD {}"
                    if exists
                    else "CREATE ROLE {} LOGIN PASSWORD {}"
                ).format(sql.Identifier(schema), sql.Literal(password))
            )
            connection.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(connection.info.dbname), sql.Identifier(schema)
                )
            )
            connection.execute(
                sql.SQL("GRANT EXECUTE ON PROCEDURE public.seed_exercise() TO {}").format(
                    sql.Identifier(schema)
                )
            )
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
            connection.execute(
                sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                    sql.Identifier(schema), sql.Identifier(schema)
                )
            )
            connection.execute(
                sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(schema))
            )
            connection.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(schema)))
            connection.execute("CALL public.seed_exercise()")

        self.events[session_id] = []

        return schema

    def _session_password(self, session_id: str) -> str:
        token = os.getenv("LAB3_TOOL_TOKEN", "")

        if not token:
            raise ValueError("Set LAB3_TOOL_TOKEN before creating an isolated session")

        return hmac.new(token.encode(), session_id.encode(), hashlib.sha256).hexdigest()

    def _connect(self, session_id: str) -> psycopg.Connection:
        if not self.settings.database_url:
            raise ValueError("Set LAB3_DATABASE_URL in .env and run lab3 reset --session ...")

        connection = psycopg.connect(
            make_conninfo(
                self.settings.database_url,
                user=session_schema(session_id),
                password=self._session_password(session_id),
            )
        )
        connection.execute(
            sql.SQL("SET search_path TO {}, pg_catalog").format(
                sql.Identifier(session_schema(session_id))
            )
        )
        connection.execute(
            sql.SQL("SET statement_timeout TO {}").format(
                sql.Literal(self.settings.tools["statement_timeout_ms"])
            )
        )

        return connection

    def execute_sql(self, query: str, session_id: str, explain: bool = False) -> dict:
        validate_sql(query, explain)

        with self._connect(session_id) as connection:
            cursor = connection.execute("EXPLAIN (FORMAT JSON) " + query if explain else query)
            columns = [c.name for c in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(self.settings.tools["max_rows"] + 1) if columns else []

            return {
                "columns": columns,
                "rows": rows[: self.settings.tools["max_rows"]],
                "row_count": cursor.rowcount,
                "command": cursor.statusmessage,
                "truncated": len(rows) > self.settings.tools["max_rows"],
            }

    def inspect_schema(self, session_id: str) -> dict:
        with self._connect(session_id) as connection:
            columns = connection.execute(
                "SELECT table_name,column_name,data_type,is_nullable,column_default,is_identity "
                "FROM information_schema.columns WHERE table_schema=%s "
                "ORDER BY table_name,ordinal_position",
                (session_schema(session_id),),
            ).fetchall()
            constraints = connection.execute(
                "SELECT c.relname, con.conname, pg_get_constraintdef(con.oid) "
                "FROM pg_constraint con JOIN pg_class c ON c.oid=con.conrelid "
                "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s "
                "ORDER BY c.relname,con.conname",
                (session_schema(session_id),),
            ).fetchall()

        return {
            "schema": "public (isolated exercise)",
            "columns": columns,
            "constraints": constraints,
        }

    async def call(
        self,
        name: str,
        arguments: dict,
        session_id: str,
        call_id: str | None = None,
        transport: str = "native",
    ) -> dict:
        started = time.perf_counter()
        identifier = call_id or uuid.uuid4().hex
        result = ToolResult(call_id=identifier, tool_name=name, status="ok")

        try:
            if name not in ARGUMENTS:
                raise ValueError(f"Unknown tool {name}")

            args = ARGUMENTS[name].model_validate(arguments).model_dump()

            with self.tracer.span(
                name,
                kind="tool",
                input=args,
                session_id=session_id,
                transport=transport,
                mcp_server="lab3-postgres" if transport == "mcp" else None,
                **self.turns.get(session_id, {}),
            ) as observation:
                if name == "search_docs":
                    with self.tracer.span("retrieval", kind="retriever", input=args) as retrieval:
                        result.data = await asyncio.to_thread(self.retriever.search, **args)

                        if retrieval:
                            retrieval.update(
                                output=result.data,
                                metadata={
                                    "source": self.settings.retrieval["corpus_url"],
                                    "chunks": [
                                        {
                                            k: chunk[k]
                                            for k in ("chunk_id", "document_id", "page", "score")
                                        }
                                        for chunk in result.data
                                    ],
                                },
                            )
                elif name == "read_doc_page":
                    result.data = self.retriever.read_page(**args)
                elif name == "inspect_schema":
                    result.data = await asyncio.to_thread(self.inspect_schema, session_id)
                else:
                    result.data = await asyncio.to_thread(
                        self.execute_sql, args["sql"], session_id, name == "explain_sql"
                    )

                if observation:
                    observation.update(output=result.data)
        except Exception as error:
            result.status = "error"
            result.error = {
                "code": getattr(error, "sqlstate", None) or type(error).__name__,
                "message": str(error),
                "retryable": isinstance(error, psycopg.Error),
            }

        result.duration_ms = (time.perf_counter() - started) * 1000
        result.truncated = isinstance(result.data, dict) and result.data.get("truncated", False)
        directory = self.settings.sessions / "tool_results"
        directory.mkdir(parents=True, exist_ok=True)
        result.artifact_id = uuid.uuid4().hex
        payload = (
            json.loads(result.model_dump_json())
            if result.data is None
            else json.loads(json.dumps(result.model_dump(), ensure_ascii=False, default=str))
        )
        (directory / f"{result.artifact_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        event = {**payload, "session_id": session_id, **self.turns.get(session_id, {})}
        self.events.setdefault(session_id, []).append(event)
        self.tracer.event("tool_result", **event)

        return payload


def create_tool_app(settings: Settings) -> FastAPI:
    service = ToolService(settings)
    mcp = FastMCP(
        "lab3-postgres", stateless_http=True, json_response=True, streamable_http_path="/"
    )
    descriptions = settings.prompts()["tools"]

    @mcp.tool(description=descriptions["search_docs"], structured_output=True)
    async def search_docs(query: str, ctx: Context, limit: int = 5) -> dict[str, Any]:
        return await service.call(
            "search_docs",
            {"query": query, "limit": limit},
            ctx.request_context.request.headers.get("x-lab3-session", "demo"),
            transport="mcp",
        )

    @mcp.tool(description=descriptions["read_doc_page"], structured_output=True)
    async def read_doc_page(
        document_id: str, ctx: Context, offset: int = 0, limit: int = 6000
    ) -> dict[str, Any]:
        return await service.call(
            "read_doc_page",
            {
                "document_id": document_id,
                "offset": offset,
                "limit": limit,
            },
            ctx.request_context.request.headers.get("x-lab3-session", "demo"),
            transport="mcp",
        )

    @mcp.tool(description=descriptions["inspect_schema"], structured_output=True)
    async def inspect_schema(ctx: Context, schema: str = "public") -> dict[str, Any]:
        return await service.call(
            "inspect_schema",
            {"schema": schema},
            ctx.request_context.request.headers.get("x-lab3-session", "demo"),
            transport="mcp",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            yield

        service.tracer.flush()

    app = FastAPI(lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = os.getenv("LAB3_TOOL_TOKEN", "")
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ")

        if not expected or not hmac.compare_digest(expected, supplied):
            return JSONResponse({"detail": "Set LAB3_TOOL_TOKEN and supply Bearer token"}, 401)

        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "tools": list(ARGUMENTS)}

    @app.get("/schemas")
    async def schemas():
        return service.schemas()

    @app.post("/sessions/{session_id}/reset")
    async def reset(session_id: str):
        return {"schema": await asyncio.to_thread(service.reset, session_id)}

    @app.post("/sessions/{session_id}/turn")
    async def set_turn(session_id: str, request: Request):
        service.turns[session_id] = await request.json()

        return {"ok": True}

    @app.get("/sessions/{session_id}/events")
    async def events(session_id: str):
        return service.events.get(session_id, [])

    @app.post("/tools/{name}")
    async def call_tool(name: str, request: Request):
        if name not in ARGUMENTS:
            raise HTTPException(404, "Unknown tool")

        body = await request.json()

        return await service.call(
            name,
            body.get("arguments", {}),
            request.headers.get("x-lab3-session", "demo"),
            body.get("call_id"),
        )

    app.mount("/mcp", mcp.streamable_http_app())

    return app
