import ib_insync

ib = ib_insync.IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = ib_insync.Option('QQQ', )