# import base64
# import json
#
# from bytewax.dataflow import Dataflow
# from bytewax.inputs import ManualInputConfig, distribute
#
#
# from websocket import create_connection
#
# ## input
# ticker_list = ['AMZN', 'MSFT']
#
# def yf_input(worker_tickers, state):
#         ws = create_connection("wss://streamer.finance.yahoo.com/")
#         ws.send(json.dumps({"subscribe": worker_tickers}))
#         while True:
#             yield state, ws.recv()
#
#
# def input_builder(worker_index, worker_count, resume_state):
#     state = resume_state or None
#     worker_tickers = list(distribute(ticker_list, worker_index, worker_count))
#     print({"subscribing to": worker_tickers})
#     return yf_input(worker_tickers, state)
#
#
# flow = Dataflow()
# flow.input("input", ManualInputConfig(input_builder))

import yfinance as yf

data = yf.download("AAPL", period="5d",interval="1m")
print(data.head())