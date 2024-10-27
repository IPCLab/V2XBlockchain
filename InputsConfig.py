class InputsConfig:
    ''' Input configurations for Ethereum model '''

    ''' Block Parameters '''
    # Binterval = 12.42  # Average time (in seconds)for creating a block in the blockchain
    Binterval = 2.7
    Bsize = 1.0  # The block size in MB
    # Blimit = 8000000  # The block gas limit
    Blimit = 30000000 # found at www.etherscan.io/chart
    # Bdelay = 2.3  # average block propogation delay in seconds, #Ref: https://bitslog.wordpress.com/2016/04/28/uncle-mining-an-ethereum-consensus-protocol-flaw/
    Bdelay = 0.25
    Breward = 2  # Reward for mining a block

    ''' Transaction Parameters '''
    hasTrans = True  # True/False to enable/disable transactions in the simulator
    Ttechnique = "Light"  # Full/Light to specify the way of modelling transactions
    # Tn = 20  # The rate of the number of transactions to be created per second
    # Tn = 4000
    # The average transaction propagation delay in seconds (Only if Full technique is used)
    Tdelay = 3
    # The transaction fee in Ethereum is calculated as: UsedGas X GasPrice
    Tsize = 0.000546  # The average transaction size  in MB

    ''' Drawing the values for gas related attributes (UsedGas and GasPrice, CPUTime) from fitted distributions '''

    ''' Uncles Parameters '''
    hasUncles = False  # boolean variable to indicate use of uncle mechansim or not
    Buncles = 2  # maximum number of uncle blocks allowed per block
    Ugenerations = 7  # the depth in which an uncle can be included in a block
    Ureward = 0
    UIreward = Breward / 32  # Reward for including an uncle

    ''' Node Parameters '''
    # Nn = 3  # the total number of nodes in the network
    # NODES = []
    # # from Models.Ethereum.Node import Node
    # here as an example we define three nodes by assigning a unique id for each one + % of hash (computing) power
    # NODES = [Node(id=0, hashPower=50), Node(
    #     id=1, hashPower=20), Node(id=2, hashPower=30)]

    ''' Simulation Parameters '''
    simTime = 500  # the simulation length (in seconds)
    Runs = 2  # Number of simulation runs