REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE CONNECT ON DATABASE lab3 FROM PUBLIC;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'langfuse') THEN
        CREATE ROLE langfuse LOGIN PASSWORD 'langfuse_local'
            NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END;
$$;

SELECT 'CREATE DATABASE langfuse OWNER langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')
\gexec

REVOKE ALL ON DATABASE langfuse FROM PUBLIC;
GRANT CONNECT ON DATABASE langfuse TO langfuse;

CREATE OR REPLACE PROCEDURE public.seed_exercise()
LANGUAGE plpgsql
SECURITY INVOKER
AS $seed$
BEGIN
    -- Таблицы принадлежат вызывающей роли и создаются в её текущей схеме.
    CREATE TABLE customers (
        id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        name text NOT NULL,
        city text NOT NULL,
        email text UNIQUE
    );

    CREATE TABLE products (
        id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        name text NOT NULL,
        price numeric(10,2) NOT NULL CHECK (price >= 0),
        stock integer NOT NULL CHECK (stock >= 0)
    );

    CREATE TABLE orders (
        id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        customer_id integer NOT NULL REFERENCES customers(id),
        status text NOT NULL DEFAULT 'new' CHECK (status IN ('new','paid','cancelled')),
        created_at date NOT NULL
    );

    CREATE TABLE order_items (
        order_id integer NOT NULL REFERENCES orders(id),
        product_id integer NOT NULL REFERENCES products(id),
        quantity integer NOT NULL CHECK (quantity > 0),
        PRIMARY KEY (order_id, product_id)
    );

    INSERT INTO customers (name,city,email) VALUES
    ('Анна','Москва','anna@example.test'),
    ('Борис','Казань','boris@example.test'),
    ('Вера','Москва',NULL),
    ('Глеб','Казань',NULL);

    INSERT INTO products (name,price,stock) VALUES
    ('Клавиатура',1000,10),('Мышь',500,20),('Монитор',12000,5),('Кабель',200,0);

    INSERT INTO orders (customer_id,status,created_at) VALUES
    (1,'paid','2026-01-10'),(2,'new','2026-01-11'),(1,'paid','2026-02-01'),
    (3,'cancelled','2026-02-02');

    INSERT INTO order_items VALUES (1,1,2),(1,2,1),(2,3,1),(3,2,3),(4,4,2);
END;
$seed$;

REVOKE ALL ON PROCEDURE public.seed_exercise() FROM PUBLIC;

DO $$
DECLARE
    task_role record;
BEGIN
    FOR task_role IN SELECT rolname FROM pg_roles WHERE rolname ~ '^task_[0-9a-f]{20}$'
    LOOP
        EXECUTE format('GRANT CONNECT ON DATABASE lab3 TO %I', task_role.rolname);
        EXECUTE format('GRANT EXECUTE ON PROCEDURE public.seed_exercise() TO %I', task_role.rolname);
    END LOOP;
END;
$$;
