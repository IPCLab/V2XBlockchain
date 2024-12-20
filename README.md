# A Multi-Layer Blockchain Simulator and Performance Evaluation of Social Internet of Vehicles with Multi-Connectivity Management

This system proposes a reference multi-connectivity management method to address the integration challenges between blockchain and V2X communication networks. It focuses on enhancing the success rate of retransmitted blockchain-related messages.

The multi-layer integration of blockchain technology and V2X communication is described as below.

<div align="center">
    <img src="./assets/system.jpg?raw=true" width=700>
</div>

1. The **Transportation layer** utilizes customizable vehicle datasets, such as those generated by **SUMO**, for adaptable simulations. It categorizes traffic data into urban, suburban, and rural densities, optionally using satellite data to enhance V2X connectivity. Road-side units (RSUs) are strategically positioned to enable flexible communication and function as blockchain nodes.
2. The **Connection Layer** enables V2X communication, facilitating interactions among vehicles, infrastructure, and satellites. SIoV enhances this by forming dynamic networks for sharing safety and traffic data, optimizing communication and resource use. It ensures secure message transmission and verifies each message for blockchain transactions, enhancing data integrity and reliability.
3. The **Consensus Layer** uses **BlockSim** to simulate a blockchain environment. RSUs act as nodes, packaging transactions, generating blocks, and managing broadcasting and synchronization. They compete to create blocks based on their hash power, reflecting their computational capability.

The preprint of this work is available at: https://arxiv.org/abs/2411.14000

## Installation

1. Clone the repository of this project:

```
git clone https://github.com/IPCLab/V2XBlockchain.git
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
