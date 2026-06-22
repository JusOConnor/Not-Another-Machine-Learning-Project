import datetime

class MLConfig:
    # def __init__(self, ticker, start_date, end_date):
    #     # The three you set
    #     self.ticker     = ticker
    #     self.start_date = start_date
    #     self.end_date   = end_date
    def __init__(self, ticker, start_date="2018-01-01", end_date=None):
        self.ticker     = ticker
        self.start_date = start_date
        self.end_date   = end_date or datetime.date.today().isoformat()

        # Everything else derived automatically
        self.features_path  = f'data/features_{ticker}.csv'
        self.output_path    = f'data/prices_{ticker}.csv'
        self.models_dir     = 'models'
        self.backup_dir     = 'models/backup'
        self.test_set_path  = f'data/test_set_{ticker}.csv'
        self.train_fraction = 0.80
        self.results_log    = f'reports/results_log_{ticker}.csv'
        self.best_model_ptr = f'models/best_model_{ticker}.txt'
        self.lookback_days  = 80
        self.min_improvement = 0.0