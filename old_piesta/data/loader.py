import pandas as pd
from typing import Callable, List, Union
from piesta.data.asset import Universe, UniverseTree
import sqlite3
import mysql.connector


class Loader:
    def __init__(self, universe: Union[Universe, UniverseTree], database_connection: Callable):
        self.universe = universe
        self.database_connection = database_connection

    def load_data(self, tickers: List[str]) -> pd.DataFrame:
        data_frames = []
        for ticker in tickers:
            data = self.database_connection(ticker)
            data_frames.append(data)

        result = pd.concat(data_frames, axis=1)
        return result

# Example usage:
def mock_database_connection(ticker: str) -> pd.DataFrame:
    import numpy as np
    date_range = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    data = np.random.rand(len(date_range))
    return pd.DataFrame({ticker: data}, index=date_range)

def csv_database_connection(ticker: str) -> pd.DataFrame:
    file_path = f"data/{ticker}.csv"
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return data

class SQLDatabaseConnection:
    def __init__(self, db_type: str = "sqlite3", **kwargs):
        self.db_type = db_type
        if db_type == "sqlite3":
            self.conn = sqlite3.connect(kwargs.get("db_path", "example.db"))
        elif db_type == "mysql":
            self.conn = mysql.connector.connect(
                host=kwargs.get("host", "localhost"),
                user=kwargs.get("user", "root"),
                password=kwargs.get("password", ""),
                database=kwargs.get("database", "example")
            )
        else:
            raise ValueError("Unsupported database type")
    
    def fetch_data(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        data = {}
        for ticker in tickers:
            query = f"SELECT * FROM {kwargs.get('table', 'prices')} WHERE ticker = '{ticker}'"
            ticker_data = pd.read_sql_query(query, self.conn, index_col=kwargs.get("index_col", "date"), parse_dates=True)
            data[ticker] = ticker_data
        
        return data
    
    def close(self):
        self.conn.close()


"""
# Example usage:
universe = Universe()  # or UniverseTree()
database_connection = SQLDatabaseConnection(db_type="sqlite3", db_path="example.db", table="prices", index_col="date")
loader = Loader(universe, database_connection.fetch_data)
tickers = universe.get_last_layer()
data = loader.load_data(tickers)
print(data)

database_connection.close()

#loader = Loader(universe, mock_database_connection)
"""