from InputsConfig import InputsConfig as p
import random
from Models.Block import Block
from Event import Event, Queue
from Models.Ethereum.Block import Block

class Scheduler:

    # Schedule a block creation event for a miner and add it to the event list
    def create_block_event(RSUnode, blockTime, start_time):
        # prepare attributes for the event
        block = Block()            
        block.depth = len(RSUnode.localBlockchain)         
        block.id = random.randrange(100000000000)
        block.previous = RSUnode.last_block().id
        block.start_time = start_time
        block.timestamp = blockTime
        block.miner = RSUnode.id        

        # print(f"Block ID: {block.id}, Block PoW Time: {block.timestamp}")

        event = Event("create_block", block.miner, blockTime, block)  # create the event
        Queue.add_event(event)  # add the event to the queue

    # Schedule a block receiving event for a node and add it to the event list
    def receive_block_event(recipient, block, blockDelay, count_recevie_event_times):
        receive_block_time = block.timestamp + blockDelay

        e = Event("receive_block", recipient.id, receive_block_time, block)
        Queue.add_event(e)