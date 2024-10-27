from InputsConfig import InputsConfig as p
from Models.Ethereum.Consensus import Consensus as c
from Models.Incentives import Incentives
import pandas as pd
import numpy as np

# 這裡會發生cicular import的問題，我先註解起來
# from Env import Environ


class Statistics:

    ########################################################### Global variables used to calculate and print simuation results ###########################################################################################
    def __init__(self, step, BSNodesList):
        self.totalBlocks = 0
        self.mainBlocks = 0
        self.totalUncles = 0
        self.uncleBlocks = 0
        self.staleBlocks = 0
        self.uncleRate = 0
        self.staleRate = 0
        self.blockData = []
        self.blocksResults = []
        # self.profits= [[0 for x in range(7)] for y in range(step * len(BSNodesList))] # rows number of miners * number of runs, columns =7
        self.profits = np.zeros((step * len(BSNodesList), 7))
        self.index = 0
        self.chain = []

    def calculate(self, step, BSNodesList, consensus):
        self.Global_Chain(consensus) # print the global chain
        self.blocks_results(consensus) # calcuate and print block statistics e.g., # of accepted blocks and stale rate etc
        self.profit_results(step, BSNodesList) # calculate and distribute the revenue or reward for miners

    ########################################################### Calculate block statistics Results ###########################################################################################
    def blocks_results(self, consensus):
        trans = 0

        self.mainBlocks = len(consensus.global_chain) - 1
        self.staleBlocks = self.totalBlocks - self.mainBlocks
        for b in consensus.global_chain:
            self.uncleBlocks += len(b.uncles) 
            trans += len(b.transactions)
        self.staleRate = round(self.staleBlocks/self.totalBlocks * 100, 2)
        self.uncleRate = round(self.uncleBlocks/self.totalBlocks * 100, 2) 
        self.blockData = [ self.totalBlocks, self.mainBlocks,  self.uncleBlocks, self.uncleRate, self.staleBlocks, self.staleRate, trans]
        self.blocksResults += [self.blockData]

    ########################################################### Calculate and distibute rewards among the miners ###########################################################################################
    def profit_results(self, step, BSNodesList):

        for m in BSNodesList:            
            i = self.index + int(''.join(filter(str.isdigit, m.id))) * step
            self.profits[i][0] = m.id
            self.profits[i][1] = m.hashPower
            self.profits[i][2] = m.blocks
            self.profits[i][3] = round(m.blocks/self.mainBlocks * 100,2)
            self.profits[i][4] = m.uncles
            self.profits[i][5] = round((m.blocks + m.uncles)/(self.mainBlocks + self.totalUncles) * 100,2)
            self.profits[i][6] = m.balance
        self.index += 1

    ########################################################### prepare the global chain  ###########################################################################################
    def Global_Chain(self, consensus):
        # print("Running global chain")
        for i in consensus.global_chain:
            block = [i.depth, i.id, i.previous, i.timestamp, i.miner, len(i.transactions), i.usedgas, len(i.uncles)]
            self.chain += [block]
            # print(f"global chain 第{i}個 block:{self.chain[i].id}")

    ########################################################### Print simulation results to Excel ###########################################################################################
    def print_to_excel(self, fname, BSNodesList):
        df1 = pd.DataFrame({'Block Time': [p.Binterval], 'Block Propagation Delay': [p.Bdelay], 'No. Miners': [len(BSNodesList)], 'Simulation Time': [p.simTime]})
        #data = {'Stale Rate': Results.staleRate,'Uncle Rate': Results.uncleRate ,'# Stale Blocks': Results.staleBlocks,'# Total Blocks': Results.totalBlocks, '# Included Blocks': Results.mainBlocks, '# Uncle Blocks': Results.uncleBlocks}

        df2 = pd.DataFrame(self.blocksResults)
        df2.columns = ['Total Blocks', 'Main Blocks', 'Uncle blocks', 'Uncle Rate', 'Stale Blocks', 'Stale Rate', '# transactions']

        df3 = pd.DataFrame(self.profits)
        df3.columns = ['Miner ID', '% Hash Power','# Mined Blocks', '% of main blocks','# Uncle Blocks','% of uncles', 'Profit (in ETH)']

        # df4 = pd.DataFrame(self.chain)
        # df4.columns = ['Block Depth', 'Block ID', 'Previous Block', 'Block Timestamp', 'Miner ID', '# transactions','Block Limit', 'Uncle Blocks']

        writer = pd.ExcelWriter(fname, engine = 'xlsxwriter')
        df1.to_excel(writer, sheet_name = 'InputConfig')
        df2.to_excel(writer, sheet_name = 'SimOutput')
        df3.to_excel(writer, sheet_name = 'Profit')
        # df4.to_excel(writer,sheet_name = 'Chain')

        writer._save()

    ########################################################### Reset all global variables used to calculate the simulation results ###########################################################################################
    def reset(self):
        self.totalBlocks = 0
        self.totalUncles = 0
        self.mainBlocks = 0
        self.uncleBlocks = 0
        self.staleBlocks = 0
        self.uncleRate = 0
        self.staleRate = 0
        self.blockData = []

    def reset2(self, step, BSNodesList):
        self.blocksResults = []
        self.profits = [[0 for x in range(7)] for y in range(step * len(BSNodesList))] # rows number of miners * number of runs, columns =7
        self.index = 0
        self.chain = []
