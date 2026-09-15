import argparse
import sys
from pathlib import Path

from loguru import logger

from src.main import main
from src.module.settings import settings

logger.remove()
logger.add(sys.stderr, colorize=True, level=settings.log_level)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate the monthly Inquiry Score v4 performance report.')
    parser.add_argument('--report-month', help='yyyymm label; defaults to REPORT_MONTH or current month')
    parser.add_argument('--input-csv', type=Path, help='Build the report from a CSV export of the aged pool instead of Oracle')
    parser.add_argument('--output', type=Path, help='Output HTML path; defaults to REPORT_DIR/inquiry_score_v4_report_<yyyymm>.html')
    parser.add_argument('--skip-sql', action='store_true', help='Do not rebuild aged pool tables (overrides RUN_TARGET_SQL)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(report_month=args.report_month, input_csv=args.input_csv, output=args.output, skip_sql=args.skip_sql)
