import pandas as pd

class Loader:
    def __init__(self, csv_file: str):
        """
        Loader class는 객체 생성 시 csv_file의 이름을 전달받아 각 모듈이 요청하는 시계열 데이터프레임을 반환한다.

        Attributes:
        - _read_csv(self): csv 파일 전체를 읽어 데이터프레임을 반환한다.
        - load_data(self, start_date: str, end_date: str): start/end date을 전달받아 해당 기간동안의 시계열 데이터프레임을 반환한다.
        - get_prices(self, assets: list = None): List로 전달받은 Asset(Columns)의 시계열 데이터프레임을 반환한다.
        """

        self.csv_file = csv_file
        self.prices = self._read_csv()

    def _read_csv(self):
        prices = pd.read_csv(self.csv_file, index_col=0, parse_dates=True)
        return prices

    def load_data(self, start_date: str, end_date: str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        return self.prices.loc[start_date:end_date]

    def get_prices(self, assets: list = None):
        if assets is not None:
            return self.prices[assets]
        return self.prices
