

EVOXBENCH_DATABASE_PATH = "#Path To#/evoxbench/database20220713/database"
EVOXBENCH_DATA_PATH = "#Path To#/evoxbench/data20221028/data"


if __name__ == '__main__':
    # setup before first run
    from evoxbench.database.init import config
    config(EVOXBENCH_DATABASE_PATH, EVOXBENCH_DATA_PATH)