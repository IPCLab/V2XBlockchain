from Models.Ethereum.Distribution.DistFit import DistFit
import random
from InputsConfig import InputsConfig as p
import numpy as np
from Models.Network import Network
import operator
from Models.Ethereum.Distribution.DistFit import DistFit

class Transaction(object):

    """ Defines the Ethereum Block model.

    :param int id: the uinque id or the hash of the transaction
    :param int timestamp: the time when the transaction is created. In case of Full technique, this will be array of two value (transaction creation time and receiving time)
    :param int sender: the id of the node that created and sent the transaction
    :param int to: the id of the recipint node
    :param int value: the amount of cryptocurrencies to be sent to the recipint node
    :param int size: the transaction size in MB
    :param int gasLimit: the maximum amount of gas units the transaction can use. It is specified by the submitter of the transaction
    :param int usedGas: the amount of gas used by the transaction after its execution on the EVM
    :param int gasPrice: the amount of cryptocurrencies (in Gwei) the submitter of the transaction is willing to pay per gas unit
    :param float fee: the fee of the transaction (usedGas * gasPrice)
    """

    def __init__(self, id=0, timestamp=0 or [], sender = None, to = None, value = None, size=0.000546,
                 gasLimit= 8000000, usedGas=0, gasPrice=0, fee=0):
        self.id = id
        self.timestamp = timestamp
        self.sender = sender
        self.to = to
        self.value = value
        self.size = size        
        self.gasLimit = gasLimit
        self.usedGas = usedGas
        self.gasPrice = gasPrice
        self.fee = usedGas * gasPrice

class LightTransaction():

    pool = [] # shared pool of pending transactions
    pool_with_gas = [] # saving transactions with gas computed
    v2v_tx_pool = {}
    v2s_tx_pool = {}

    def create_transactions(global_clock, transmitted_clock, VehID, ReceiverID, transmitted_data):                         
        tx = Transaction()
        tx.id = random.randrange(100000000000)
        tx.timestamp = transmitted_clock
        tx.sender = VehID
        # vehIdx = int(''.join(filter(str.isdigit, VehID)))
        # tx.sender = vehIdx
        tx.to = ReceiverID
        tx.value = transmitted_data        

        # print(f"TX Timestamp: {tx.timestamp}")
        # print(f"發送車輛: {tx.sender}, 接收者: {tx.to}")
        
        if ReceiverID[:3] == "veh": # 當交易走V2V通道時
            if ReceiverID not in LightTransaction.v2v_tx_pool.keys(): # 如果v2v_tx_pool的key不存在交易接收車輛
                LightTransaction.v2v_tx_pool[ReceiverID] = [] # 為這個交易接收車輛建立一個子字典
            LightTransaction.v2v_tx_pool[ReceiverID].append(tx) # 該筆交易先暫存於v2v_tx_pool
            # print(f"發送車輛: {tx.sender}, 接收者: {tx.to}")
            # print(f"成功將V2V交易 {tx.id} 加進車輛的pool")
            # print(f"成功將V2V交易 {tx.value} 加進車輛的pool")
        elif ReceiverID[:3] == "sat": # 當交易走V2S通道時
            if tx.timestamp not in LightTransaction.v2s_tx_pool.keys(): # 如果v2s_tx_pool的key不存在此時的global_clock
                LightTransaction.v2s_tx_pool[tx.timestamp] = [] # 為這個global_clock建立一個子字典
            LightTransaction.v2s_tx_pool[tx.timestamp].append(tx) # 將在此刻global_clock的V2S交易先暫存於v2s_tx_pool
            # print(f"成功將V2S交易 {tx.id} 加進衛星的pool")
            # print(f"成功將V2S交易 {tx.value} 加進衛星的pool")
        else: # 當交易走V2I通道時
            LightTransaction.pool.append(tx) # 該筆交易直接放入pool
            # print(f"成功將V2I交易 {tx.id} 加進RSU的pool")
            # print(f"成功將V2I交易 {tx.value} 加進RSU的pool")

        # if RecriverID[:3] == "rsu": # 當交易走V2I通道時，檢查是否有之前來自V2V的交易暫存於發送車輛上
        #     idx = 0            
        #     while VehID in LightTransaction.v2v_tx_pool.keys() and idx < len(LightTransaction.v2v_tx_pool[VehID]):
        #         # 檢查交易發送車輛是否存在於v2v_tx_pool
        #         tx_in_veh = LightTransaction.v2v_tx_pool[VehID][idx] 
        #         if tx.timestamp > tx_in_veh.timestamp: # 現在的時刻需要大於暫存於v2v_tx_pool交易的時刻
        #             LightTransaction.pool.append(tx_in_veh) # 將這筆交易放進pool
        #             del LightTransaction.v2v_tx_pool[VehID][idx] # 將這筆交易從v2v_tx_pool刪除
        #             print(f"成功將V2V2I交易 {tx_in_veh.id} 加進RSU的pool")
        #         else: 
        #             idx += 1

        # del_key = []
        # # if global_clock-0.024 in LightTransaction.v2s_tx_pool.keys(): # 如果有經過24ms的交易存於v2s_tx_pool中
        # for key in LightTransaction.v2s_tx_pool.keys():
        #     if key+0.024 <= global_clock and key+0.024 > global_clock-0.005:
        #     # for tx_in_sat in LightTransaction.v2s_tx_pool[global_clock-0.024]: # 因為V2S和S2I的傳輸延遲各為12ms
        #         for tx_in_sat in LightTransaction.v2s_tx_pool[key]:
        #             LightTransaction.pool.append(tx_in_sat) # 將交易一個個放進pool
        #         # print(f"成功將V2S2I交易 {tx_in_sat.id} 加進RSU的pool")
        #         # del LightTransaction.v2s_tx_pool[global_clock-0.024] # 將有經過24ms的交易從v2s_tx_pool刪除
        #         del_key.append(key)
        # for key in del_key:
        #     del LightTransaction.v2s_tx_pool[key]
    
        # print(f"Transaction ID: {tx.id}, Transaction Timestamp: {tx.timestamp}, Transaction Sender: {tx.sender}, Transaction Receiver: {tx.to}, Transaction value: {tx.value}")

        random.shuffle(LightTransaction.pool) 
    
    def clean_pool():
        LightTransaction.pool = [] 
        LightTransaction.pool_with_gas = [] 
        LightTransaction.v2v_tx_pool = {}
        LightTransaction.v2s_tx_pool = {}
    
    def calculateGas():
        Psize = len(LightTransaction.pool)
        # Psize = int(p.Tn * p.Binterval) # 248
        print(f"Psize = {Psize}")

        DistFit.fit() # fit distributions
        gasLimit,usedGas,gasPrice,_ = DistFit.sample_transactions(Psize)
        # print(f"calculateGas 交易總數: {len(LightTransaction.pool)}")

        # for idx, tx in enumerate(LightTransaction.pool[:Psize]):
        for idx, tx in enumerate(LightTransaction.pool):
            # vehID_digits = ''.join(c for c in tx.sender if c.isdigit()) 
            # vehIdx = int(vehID_digits)
            # print(vehIdx)
            tx.gasLimit = gasLimit[idx]
            tx.usedGas = usedGas[idx]
            tx.gasPrice = gasPrice[idx]/1000000000
            tx.fee = tx.usedGas * tx.gasPrice
            LightTransaction.pool_with_gas.append(tx)
            
        # remove transactions from original pool 
        # LightTransaction.pool = LightTransaction.pool[Psize:]
        LightTransaction.pool = []
        
    ##### Select and execute a number of transactions to be added in the next block #####
    def execute_transactions():
        # from pool with gas 
        count = 0 
        transactions = [] # prepare a list of transactions to be included in the block
        blocklimit = p.Blimit # Blimit = 8000000 # 一個block最多可包含的gas數量
        totalGas = 0 # 統計一個block中transaction的usedGas之總量

        # sort pending transactions in the pool based on the gasPrice value
        LightTransaction.pool_with_gas = sorted(LightTransaction.pool_with_gas, key=lambda x: x.gasPrice, reverse=True) # Transaction的gasPrice表示發送者訂定一個gas的價錢
        
        # print(f"Transaction ID    Gas Limit    Used Gas    Gas Price")
        while len(LightTransaction.pool_with_gas) > count:
            if  blocklimit >= LightTransaction.pool_with_gas[count].gasLimit: # Transaction的gaslimit表示一個Transaction最多可負擔的gas數量(計算資源/成本)，一般為發送者訂定
                transactions += [LightTransaction.pool_with_gas[count]]
                blocklimit -= LightTransaction.pool_with_gas[count].usedGas # Transaction的usedGas表示這筆Transaction實際使用的gas數量
                totalGas += LightTransaction.pool_with_gas[count].usedGas
                # print(f"{LightTransaction.pool_with_gas[count].id}    {LightTransaction.pool_with_gas[count].gasLimit}    \
                #     {LightTransaction.pool_with_gas[count].usedGas}    {LightTransaction.pool_with_gas[count].gasPrice}")
                del LightTransaction.pool_with_gas[count]
            else:
                count += 1
        
        # LightTransaction.pool_with_gas.clear()

        num_trans_in_pool = len(LightTransaction.pool_with_gas)

        # return transactions, totalGas
        return transactions, totalGas, num_trans_in_pool
    
    ##### Select and execute a number of transactions to be added in the next block #####
    def execute_transactions_original():
        count = 0 
        transactions = [] # prepare a list of transactions to be included in the block
        blocklimit = p.Blimit # Blimit = 8000000 # 一個block最多可包含的gas數量
        totalGas = 0 # 統計一個block中transaction的usedGas之總量

        # sort pending transactions in the pool based on the gasPrice value
        LightTransaction.pool = sorted(LightTransaction.pool, key=lambda x: x.gasPrice, reverse=True) # Transaction的gasPrice表示發送者訂定一個gas的價錢
        
        print(f"Transaction ID    Gas Limit    Used Gas    Gas Price")
        while len(LightTransaction.pool) > count:
            if  blocklimit >= LightTransaction.pool[count].gasLimit: # Transaction的gaslimit表示一個Transaction最多可負擔的gas數量(計算資源/成本)，一般為發送者訂定
                # print(f"Transaction ID: {LightTransaction.pool[count].id}") 
                # print(f"Transaction Gas Limit: {LightTransaction.pool[count].gasLimit}")
                transactions += [LightTransaction.pool[count]]
                blocklimit -= LightTransaction.pool[count].usedGas # Transaction的usedGas表示這筆Transaction實際使用的gas數量
                totalGas += LightTransaction.pool[count].usedGas
                # print(f"Transaction Used Gas: {LightTransaction.pool[count].usedGas}")
                print(f"{LightTransaction.pool[count].id}    {LightTransaction.pool[count].gasLimit}    {LightTransaction.pool[count].usedGas}    {LightTransaction.pool[count].gasPrice}")
                del LightTransaction.pool[count]
            else:
                count += 1

        num_trans_in_pool = len(LightTransaction.pool)
        # print("execute_transactions")
        # print(f"交易池中剩下的交易數量:{len(LightTransaction.pool)}")
        print(f"被加進區塊的交易數量: {len(transactions)}")           
        print(f"Block Gas Limit: {blocklimit}")
        # return transactions, totalGas
        return transactions, totalGas, num_trans_in_pool

class FullTransaction():
    x=0 # counter to only fit distributions once during the simulation

    def create_transactions():
        Psize= int(p.Tn * p.Binterval)

        if FullTransaction.x<1:
            DistFit.fit() # fit distributions
        gasLimit,usedGas,gasPrice,_ = DistFit.sample_transactions(Psize) # sampling gas based attributes for transactions from specific distribution

        for i in range(Psize):
            # assign values for transactions' attributes. You can ignore some attributes if not of an interest, and the default values will then be used
            tx= Transaction()

            tx.id= random.randrange(100000000000)
            creation_time= random.randint(0,p.simTime-1)
            receive_time= creation_time
            tx.timestamp= [creation_time,receive_time]
            sender= random.choice (p.NODES)
            tx.sender = sender.id
            tx.to= random.choice (p.NODES).id
            tx.gasLimit=gasLimit[i]
            tx.usedGas=usedGas[i]
            tx.gasPrice=gasPrice[i]/1000000000
            tx.fee= tx.usedGas * tx.gasPrice

            sender.transactionsPool.append(tx)
            FullTransaction.transaction_prop(tx)

    # Transaction propogation & preparing pending lists for miners
    def transaction_prop(tx):
        # Fill each pending list. This is for transaction propogation
        for i in p.NODES:
            if tx.sender != i.id:
                t= tx
                t.timestamp[1] = t.timestamp[1] + Network.tx_prop_delay() # transaction propogation delay in seconds
                i.transactionsPool.append(t)



    def execute_transactions(miner,currentTime):
        transactions= [] # prepare a list of transactions to be included in the block
        limit = 0 # calculate the total block gaslimit
        count=0
        blocklimit = p.Blimit
        miner.transactionsPool.sort(key=operator.attrgetter('gasPrice'), reverse=True)
        pool= miner.transactionsPool

        while count < len(pool):
                if  (blocklimit >= pool[count].gasLimit and pool[count].timestamp[1] <= currentTime):
                    blocklimit -= pool[count].usedGas
                    transactions += [pool[count]]
                    limit += pool[count].usedGas
                count+=1

        return transactions, limit
