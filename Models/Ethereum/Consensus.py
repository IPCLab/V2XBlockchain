import numpy as np
from InputsConfig import InputsConfig as p
from Models.Ethereum.Node import Node
# from Models.Consensus import Consensus as BaseConsensus
import random

#from ImportClasses import Node

class Consensus():
    
    def __init__(self):
        self.global_chain = []

    """ 
	We modelled PoW consensus protocol by drawing the time it takes the miner to finish the PoW from an exponential distribution
        based on the invested hash power (computing power) fraction
    """
    def Protocol(RSUminer, RSUNodesList):
        ##### Start solving a fresh PoW on top of last block appended #####

        TOTAL_HASHPOWER = sum([miner.hashPower for miner in RSUNodesList])
        # print(f"Total Hash Power: {TOTAL_HASHPOWER}")
                
        hashPower_rate = RSUminer.hashPower/TOTAL_HASHPOWER
        # print(f"Miner ID: {RSUminer.id}, Hash Power Rate: {hashPower_rate}")
        
        return random.expovariate(hashPower_rate * 1/p.Binterval)


    """ 
	This method apply the longest-chain approach to resolve the forks that occur when nodes have multiple differeing copies of the blockchain ledger
    """
    def fork_resolution(self, BSNodesList):
        # the accpted global chain after resovling the forks
        # self.global_chain = [] # reset the global chain before filling it

        a = []
        for BSnode in BSNodesList:
            a += [BSnode.blockchain_length()]
        x = max(a)

        b = []
        z = 0
        for BSnode in BSNodesList:
            if BSnode.blockchain_length() == x:
                b += [BSnode.id]
                z = BSnode.id

        if len(b) > 1:
            c=[]
            for BSnode in BSNodesList:
                if BSnode.blockchain_length() == x:
                    c += [int(''.join(filter(str.isdigit, BSnode.last_block().miner)))]
            z = np.bincount(c)
            z= np.argmax(z)

        # print(f"BS Node List {len(BSNodesList)}")
        for BSnode in BSNodesList:
            if BSnode.blockchain_length() == x and BSnode.last_block().miner == z:
                for bc in range(len(BSnode.localBlockchain)):
                    Consensus.global_chain.append(BSnode.localBlockchain[bc])
                    # print(f"global chain {len(Consensus.global_chain)}")
                break



