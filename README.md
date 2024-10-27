# Performance Evaluation of Blockchain-Integrated V2X Networks with Multi-Connectivity Management

This system resolves the integration challenges between blockchain and V2X communication network operating over three different areas and utilizing multiple connectivity options.

The multi-layer integration of blockchain technology and V2X communication is described as below.

<div align="center">
    <img src="./assets/system.png?raw=true" width=700>
</div>

1. The SUMO software tool is used to create vehicle datasets for simulating **Transportation Layer**, classifying traffic data into urban, suburban, and rural areas. RSUs were placed based on vehicle positions, and satellite data came from CelesTrak’s Starlink dataset. Non-mobile RSUs were designated as blockchain nodes, with each node given a node ID, transaction pool, local blockchain, and hash power, which was distributed based on a Gaussian distribution.
2. The **Connection Layer** enables comprehensive real-time V2X communication, facilitating interactions between vehicles, infrastructure, and satellites. In our proposed method, when a vehicle needs to transmit a message, it will determine its transmission method through the multi-connectivity selection process.
3. The **Consensus Layer** involves a blockchain simulation using BlockSim, where RSUs serve as blockchain nodes for transaction packaging, block creation, broadcasting, and synchronization. Nodes compete to generate blocks based on their computational power.

## Installation

1. Clone the repository of this project:

```
git clone
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

-   Run the main bash script:

```
bash run_maxsinr.sh
```
