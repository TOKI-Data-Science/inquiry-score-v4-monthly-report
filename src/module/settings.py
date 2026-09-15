"""
You can create a file called `.env` in the root of the repo, containing your local env vars.
"""
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv(verbose=True)


class Settings(BaseSettings):
    """.env variables have priority over following default values"""

    oracle_username: str
    oracle_password: str
    oracle_service: str
    oracle_hostname: str
    oracle_port: str = '1521'
    log_level: str = 'INFO'

    # report settings
    run_target_sql: bool = True  # rebuild t_inq_v4_*_aged_pool tables before reporting
    aged_pool_table: str = Field('t_inq_v4_aged_pool', pattern=r'^[A-Za-z_][A-Za-z0-9_.$]*$')
    report_dir: str = 'reports'
    report_month: str = ''  # yyyymm; empty = current month at run time
    report_start_month: int = 202504  # first base_month shown after the OOT-valid point
    score_higher_is_better: bool = True  # direction used for Gini / KS sign

    class Config:
        """.env file location"""

        env_file = ".env"


settings = Settings()
