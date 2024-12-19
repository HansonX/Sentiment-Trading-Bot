from sentiment import estimate_sentiment
from datetime import datetime, timedelta
from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from alpaca_trade_api import REST
import logging

# Alpaca API credentials
API_KEY = "API_KEY"
API_SECRET = "API_SECRET" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TradingBot(Strategy): 
    def initialize(self, symbol="SPY", cash_risk=.5, sentiment_threshold=0.99, take_profit_buy=1.25, stop_loss_buy=0.95, take_profit_sell=0.75, stop_loss_sell=1.05): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None
        self.cash_risk = cash_risk

        self.sentiment_threshold = sentiment_threshold
        self.take_profit_buy = take_profit_buy
        self.stop_loss_buy = stop_loss_buy
        self.take_profit_sell = take_profit_sell
        self.stop_loss_sell = stop_loss_sell

        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self): 
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        if last_price <= 0:
            logging.warning("Last price is non-positive, cannot compute position size.")
            return cash, last_price, 0
        
        quantity = round(cash * self.cash_risk / last_price, 0)
        if quantity < 1:
            logging.warning("Position size would be less than 1 share, no trades will be made.")
            return cash, last_price, 0
        
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        try:
            news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today) 
            headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        except Exception as e:
            logging.error(f"Error getting news: {e}")
            return 0, "neutral"
        
        if not headlines:
            logging.info("No news retrieved. Defaulting to neutral sentiment.")
            return 0, "neutral"
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment 

    def execute_trade(self, transaction, quantity, last_price):
        if transaction == "buy":
            take_profit = last_price * self.take_profit_buy
            stop_loss = last_price * self.stop_loss_buy
        else:
            take_profit = last_price * self.take_profit_sell
            stop_loss = last_price * self.stop_loss_sell
        order = self.create_order(
            self.symbol,
            quantity,
            transaction,
            type="bracket",
            take_profit_price=take_profit,
            stop_loss_price=stop_loss
        )
        try:
            self.submit_order(order)
            self.last_trade = transaction
        except Exception as e:
            logging.error(f"Failed to submit {transaction} order: {e}")

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        if quantity == 0:
            logging.info("Quantity is zero; skipping trade execution.")
            return

        probability, sentiment = self.get_sentiment()

        if cash < last_price:
            logging.info("Not enough cash to open a position.")
            return

        if sentiment == "positive" and probability >= self.sentiment_threshold:
            # If last action was a sell, close it before buying
            if self.last_trade == "sell":
                self.sell_all()  
            self.execute_trade("buy", quantity, last_price)

        elif sentiment == "negative" and probability >= self.sentiment_threshold:
            # If last action was a buy, close it before selling
            if self.last_trade == "buy":
                self.sell_all()
            self.execute_trade("sell", quantity, last_price)

        else:
            logging.info("Sentiment not strong enough; no trade executed.")

start = datetime(2021,1,1)
end = datetime(2024,11,30) 
broker = Alpaca(ALPACA_CREDS)

strategy = TradingBot(name="SentimentBot", broker=broker, parameters={"symbol":"SPY", "cash_risk":.5})
strategy.backtest(YahooDataBacktesting, start, end, parameters={"symbol":"SPY", "cash_risk":.5})