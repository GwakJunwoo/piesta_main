import datetime
import pandas as pd

class TradingDayCalendar:
    def __init__(self, start_date, end_date, holidays=[]):
        self.start_date = start_date
        self.end_date = end_date
        self.tmp_date = start_date
        self.holidays = set(holidays)

    def get_start_date(self):
        return self.start_date

    def get_end_date(self):
        return self.end_date
    
    def set_tmp_date(self, tmp_date):
        self.tmp_date = tmp_date

    def get_tmp_date(self):
        return self.tmp_date

    def is_trading_day(self, date):
        return date.weekday() < 5 and date not in self.holidays

    def get_next_trading_day(self, date):
        next_day = date + datetime.timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += datetime.timedelta(days=1)
        return next_day

    def get_previous_trading_day(self, date):
        previous_day = date - datetime.timedelta(days=1)
        while not self.is_trading_day(previous_day):
            previous_day -= datetime.timedelta(days=1)
        return previous_day

    def find_nearest_trading_days(self, date):
        previous_trading_day = self.get_previous_trading_day(date)
        next_trading_day = self.get_next_trading_day(date)
        return previous_trading_day, next_trading_day

    def count_business_days(self, start_date, end_date):
        current_date = start_date
        business_days = 0
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                business_days += 1
            current_date += datetime.timedelta(days=1)
        return business_days

    def get_month_end_trading_days(self, start_date, end_date, freq):
        current_date = start_date
        month_end_trading_days = []
        while current_date <= end_date:
            if freq == '1w':
                next_date = current_date + datetime.timedelta(weeks=1)
                next_date = self.get_previous_trading_day(next_date)
            elif freq == '2w':
                next_date = current_date + datetime.timedelta(weeks=2)
                next_date = self.get_previous_trading_day(next_date)
            elif freq == '1m':
                next_date = self.get_next_month_end(current_date)
            elif freq == '3m':
                next_date = self.get_next_month_end(current_date, months=3)
            elif freq == '6m':
                next_date = self.get_next_month_end(current_date, months=6)
            elif freq == '1y':
                next_date = self.get_next_month_end(current_date, months=12)
            else:
                raise ValueError("Invalid frequency. Please choose one of: 1w, 3w, 1m, 3m, 1y.")
            
            if next_date <= end_date:
                month_end_trading_days.append(next_date)
            
            current_date = next_date + datetime.timedelta(days=1)
        return month_end_trading_days

    def get_next_month_end(self, date, months=1):
        next_month = (date.month + months - 1) % 12 + 1
        next_year = date.year + (date.month + months - 1) // 12
        next_month_end = datetime.date(next_year, next_month, 1) - datetime.timedelta(days=1)
        return self.get_previous_trading_day(next_month_end)

    def get_all_trading_days(self, start_date, end_date):
        current_date = start_date
        trading_days = []
        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += datetime.timedelta(days=1)
        return trading_days

    def get_all_calendar_days(self, start_date, end_date):
        current_date = start_date
        calendar_days = []
        while current_date <= end_date:
            calendar_days.append(current_date)
            current_date += datetime.timedelta(days=1)
        return calendar_days

    def find_matching_dates(data_index, month_end_trading_days):
        matching_dates = []
        for date in month_end_trading_days:
            if date in data_index:
                matching_dates.append(date)
            else:
                closest_date = data_index[data_index <= date].max()
                if closest_date is not pd.NaT:
                    matching_dates.append(closest_date)
        return matching_dates

    def find_previous_trading_day(self, target_date, date_list):
        trading_days = pd.to_datetime(date_list)

        previous_day = trading_days[trading_days < target_date].max()
        return previous_day