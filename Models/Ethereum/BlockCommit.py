from Scheduler import Scheduler
from InputsConfig import InputsConfig as p
from Models.Ethereum.Node import Node
from Statistics import Statistics
from Models.Ethereum.Transaction import LightTransaction as LT, FullTransaction as FT
from Models.Network import Network
from Models.Ethereum.Consensus import Consensus as c
from Models.BlockCommit import BlockCommit as BaseBlockCommit

class BlockCommit(BaseBlockCommit):
    '''
    Handling and running Events
    def handle_event(event, RSUNodesList, statistics):
        if event.type == "create_block":
            # remain_trans = BlockCommit.generate_block(event, RSUNodesList, statistics)
            BlockCommit.generate_block(event, RSUNodesList, statistics)
        elif event.type == "receive_block":
            BlockCommit.receive_block(event, RSUNodesList)
        return remain_trans
    '''

    # Block Creation Event
    def generate_block (event, RSUNodesList, statistics):
        for RSUnode in RSUNodesList:
            if RSUnode.id == event.block.miner:
                miner = RSUnode
                break
        
        minerId = miner.id
        eventTime = event.time
        blockPrev = event.block.previous
        miner_last_block_id = miner.last_block().id
        # print(f"blockPrev: {blockPrev}")
        # print(f"miner.last_block().id: {miner.last_block().id}")

        # if blockPrev != miner_last_block_id:
        #     event.block.previous = miner_last_block_id
        #     blockPrev = event.block.previous

        if blockPrev == miner.last_block().id:
            statistics.totalBlocks += 1 # count # of total blocks created!
            
            if p.Ttechnique == "Light": 
                blockTrans, blockSize, remain_trans = LT.execute_transactions()
                # blockTrans, blockSize = LT.execute_transactions()
            elif p.Ttechnique == "Full": 
                blockTrans, blockSize = FT.execute_transactions(miner,eventTime)

            event.block.transactions = blockTrans # a list of transactions to be included in the block
            print(f"blockTrans: {len(blockTrans)}")
            event.block.usedgas = blockSize # 統計一個block中transaction的usedGas之總量

            if p.hasUncles:
                BlockCommit.update_unclechain(miner)
                blockUncles = Node.add_uncles(miner) # add uncles to the block
                event.block.uncles = blockUncles #(only when uncles activated)

            miner.localBlockchain.append(event.block)
            new_block_idex = len(miner.localBlockchain) - 1
            previous_block_idex = len(miner.localBlockchain) - 2            
            # print(f"Miner ID: {minerId}, New Block ID: {miner.localBlockchain[new_block_idex].id}")
            # print(f"Previous Block ID: {miner.localBlockchain[previous_block_idex].id}, Number of Transactions: {len(miner.localBlockchain[new_block_idex].transactions)}")
            
            '''
            if p.hasTrans and p.Ttechnique == "Light":
                LT.create_transactions() # generate transactions
            '''
            # if p.hasTrans and p.Ttechnique == "Light":LT.create_transactions()    
            BlockCommit.propagate_block(event.block, RSUNodesList)
            BlockCommit.generate_next_block(miner, eventTime, RSUNodesList)# Start mining or working on the next block

            return remain_trans

    # Block Receiving Event
    def receive_block (event, RSUNodesList):
        for RSUnode in RSUNodesList:
            if RSUnode.id == event.block.miner:
                miner = RSUnode
                break

        minerId = miner.id 
        currentTime = event.time
        blockPrev = event.block.previous # previous block id

        for RSUnode in RSUNodesList:
            if RSUnode.id == event.node:
                receiver = RSUnode
        
        receiverId = receiver.id
        lastBlockId = receiver.last_block().id # the id of last block

        #### case 1: the received block is built on top of the last block according to the recipient's blockchain ####
        if blockPrev == lastBlockId:
            receiver.localBlockchain.append(event.block) # append the block to local blockchain

            if p.hasTrans and p.Ttechnique == "Full": 
                BaseBlockCommit.update_transactionsPool(receiver, event.block)
            
            # print(f"receive_block case 1")
            BlockCommit.generate_next_block(receiver, currentTime, RSUNodesList)# Start mining or working on the next block

         #### case 2: the received block is not built on top of the last block ####
        else:
            depth = event.block.depth + 1
            if (depth > len(receiver.localBlockchain)):
                BlockCommit.update_local_blockchain(receiver, miner, depth, receiverId)
                BlockCommit.generate_next_block(receiver, currentTime, RSUNodesList)# Start mining or working on the next block

            #### 2- if depth of the received block <= depth of the last block, then reject the block (add it to unclechain) ####
            else:
                 uncle = event.block
                 receiver.unclechain.append(uncle)

            if p.hasUncles: 
                BlockCommit.update_unclechain(receiver)
            if p.hasTrans and p.Ttechnique == "Full": 
                BaseBlockCommit.update_transactionsPool(receiver, event.block) # not sure yet.

        new_block_idex = len(receiver.localBlockchain) - 1
        previous_block_idex = len(receiver.localBlockchain) - 2
        # print(f"Receiver ID: {receiverId}, New Block ID: {receiver.localBlockchain[new_block_idex].id}")
        # print(f"Previous Block ID: {receiver.localBlockchain[previous_block_idex].id}, Number of Transactions: {len(receiver.localBlockchain[new_block_idex].transactions)}")

    # Upon generating or receiving a block, the miner start working on the next block as in POW
    def generate_next_block(RSUnode, currentTime, RSUNodesList):
        if RSUnode.hashPower > 0:
            blockTime = currentTime + c.Protocol(RSUnode, RSUNodesList) # time when miner x generate the next block
            start_time = currentTime # 記錄區塊產生的起始時間
            PoW_time = blockTime - start_time
            Scheduler.create_block_event(RSUnode, blockTime, start_time)            

    def generate_initial_events(RSUNodesList, currentTime):
        for RSUnode in RSUNodesList:
            BlockCommit.generate_next_block(RSUnode, currentTime, RSUNodesList)

    def propagate_block (block, RSUNodesList):
        count_recevie_event_times = 0
        for recipient in RSUNodesList:            
            if recipient.id != block.miner:
                blockDelay = Network.block_prop_delay() # draw block propagation delay from a distribution !! or you can assign 0 to ignore block propagation delay
                # print(f"block propagation delay: {blockDelay}")
                # print(blockDelay)
                count_recevie_event_times += 1
                Scheduler.receive_block_event(recipient, block, blockDelay, count_recevie_event_times)

    def update_local_blockchain(receiver, miner, depth, receiverId):
        # the node here is the one that needs to update its blockchain, while miner here is the one who owns the last block generated
        # the node will update its blockchain to mach the miner's blockchain
        from InputsConfig import InputsConfig as p
        i = 0
        while (i < depth):
            if (i < len(receiver.localBlockchain)):
                if (receiver.localBlockchain[i].id != miner.localBlockchain[i].id): # and (self.node.blockchain[i-1].id == Miner.blockchain[i].previous) and (i>=1):
                    receiver.unclechain.append(receiver.localBlockchain[i]) # move block to unclechain
                    newBlock = miner.localBlockchain[i]
                    receiver.localBlockchain[i]= newBlock
                    if p.hasTrans and p.Ttechnique == "Full": BaseBlockCommit.update_transactionsPool(receiver, newBlock)
            else:
                newBlock = miner.localBlockchain[i]
                receiver.localBlockchain.append(newBlock)
                if p.hasTrans and p.Ttechnique == "Full": BaseBlockCommit.update_transactionsPool(receiver, newBlock)
            i += 1

    # Upon receiving a block, update local unclechain to remove all uncles included in the received block
    def update_unclechain(node):
        ### remove all duplicates uncles in the miner's unclechain
        a = set()
        x=0
        while x < len(node.unclechain):
            if node.unclechain[x].id in a:
                del node.unclechain[x]
                x-=1
            else:
                a.add(node.unclechain[x].id)
            x+=1

        j=0
        while j < len (node.unclechain):
            for k in node.blockchain:
                if node.unclechain[j].id == k.id:
                    del node.unclechain[j] # delete uncle after inclusion
                    j-=1
                    break
            j+=1

        j=0
        while j < len (node.unclechain):
            c="t"
            for k in node.blockchain:
                u=0
                while u < len(k.uncles):
                    if node.unclechain[j].id == k.uncles[u].id:
                        del node.unclechain[j] # delete uncle after inclusion
                        j-=1
                        c="f"
                        break
                    u+=1
                if c=="f":
                    break
            j+=1
