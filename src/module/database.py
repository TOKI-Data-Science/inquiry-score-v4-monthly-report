import math

import pandas as pd
from loguru import logger
from sqlalchemy import URL, create_engine, types
from sqlalchemy.exc import DatabaseError
from sqlalchemy.sql import text

from src.module.helper import logging_timer
from src.module.settings import settings

# oracle connection string for SQL alchemy engine
oracle_connection_url = URL.create(
    drivername='oracle+oracledb',
    password=settings.oracle_password,
    username=settings.oracle_username,
    port=settings.oracle_port,
    host=settings.oracle_hostname,
    query={'service_name': settings.oracle_service},
)


def col_length(str_len):
    """Calculate column lengths. Used for allocating column size"""
    # threshold = 2**(pw-1)
    pw = int(math.log(str_len, 2))
    return int(math.ceil((str_len + 2 ** (pw - 1)) / 2 ** (pw - 1)) * 2 ** (pw - 1))


def sql_col(data_frame):
    """Convert python data types to sql data types"""
    dtypes_dict = {}
    for column, dtype in zip(data_frame.columns, data_frame.dtypes):
        if "object" in str(dtype):
            str_max_len = col_length(data_frame[column].str.len().max())
            dtypes_dict.update({column: types.VARCHAR(length=str_max_len)})
        if "datetime" in str(dtype):
            dtypes_dict.update({column: types.DateTime()})
        if "float" in str(dtype):
            dtypes_dict.update({column: types.FLOAT})
        if "int" in str(dtype):
            dtypes_dict.update({column: types.INT()})
    return dtypes_dict


def sql_open(filepath):
    """Load sql file
    Do not include ; in a query
    Only one query in a file is allowed
    """
    query = open(filepath, encoding='utf-8').read()
    return query


@logging_timer()
def oracle_export(data_frame, table_name, index=False, if_exists='replace'):
    """Export to oracle DB"""
    engine = create_engine(oracle_connection_url)
    output_dtypes_dict = sql_col(data_frame)
    data_frame.to_sql(
        table_name.lower(),
        con=engine,
        if_exists=if_exists,
        index=index,
        dtype=output_dtypes_dict,
    )


@logging_timer()
def oracle_execute(query):
    """Executes query"""
    engine = create_engine(oracle_connection_url)
    with engine.connect() as connection:
        connection.execute(text(query))
        connection.commit()


@logging_timer(entry=True, exit=True)
def oracle_execute_script(filepath):
    """Executes a multi-statement sql script (statements separated by ;).
    `drop table` failures are tolerated so the script is re-runnable.
    """
    statements = [s.strip() for s in sql_open(filepath).split(';') if s.strip()]
    engine = create_engine(oracle_connection_url)
    with engine.connect() as connection:
        for statement in statements:
            first_line = statement.splitlines()[0][:80]
            try:
                connection.execute(text(statement))
                connection.commit()
                logger.info(f'Executed: {first_line}')
            except DatabaseError as err:
                if statement.lower().startswith('drop table'):
                    logger.warning(f'Skipped (table may not exist): {first_line}')
                    continue
                raise err


@logging_timer()
def oracle_import(query):
    """Import from oracle DB"""
    engine = create_engine(oracle_connection_url)
    with engine.connect() as connection:
        data_frame = pd.read_sql(text(query), connection)
    return data_frame


@logging_timer()
def oracle_sysdate(before_today=0):
    """Get current sysdate from Oracle DB"""
    query = f"SELECT TO_CHAR(SYSDATE - {before_today}, 'YYYYMMDD') FROM dual"
    return oracle_import(query).iloc[0, 0]
