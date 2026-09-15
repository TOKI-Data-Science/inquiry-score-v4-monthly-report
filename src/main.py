from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.module.helper import logging_timer
from src.module.report import generate_report
from src.module.settings import settings

TARGET_SQL = Path(__file__).parent / 'module' / 'queries' / 'target.sql'


@logging_timer(entry=True, exit=True)
def main(report_month=None, input_csv=None, output=None, skip_sql=False):
    """Monthly pipeline: (re)build aged pool tables -> load -> HTML report"""
    report_month = report_month or settings.report_month or datetime.now().strftime('%Y%m')
    logger.info(f'Report month: {report_month}')

    if input_csv:
        data = pd.read_csv(input_csv)
        logger.info(f'Loaded {len(data):,} rows from {input_csv}')
    else:
        # imported lazily so CSV runs do not need Oracle client / credentials
        from src.module.database import oracle_execute_script, oracle_import

        if settings.run_target_sql and not skip_sql:
            oracle_execute_script(TARGET_SQL)
        else:
            logger.info('Skipping target.sql')
        data = oracle_import(f'select * from {settings.aged_pool_table}')
        logger.info(f'Loaded {len(data):,} rows from {settings.aged_pool_table}')

    if data.empty:
        raise ValueError('Aged pool returned no rows')

    out_path = generate_report(data, report_month, output)
    logger.success(f'Report written: {out_path.resolve()}')
    return out_path
