from os.path import join

import json
import requests
import jsonlines

import numpy as np
import pandas as pd

from web3 import Web3, HTTPProvider
from evmdasm import EvmBytecode
from sklearn.model_selection import train_test_split

from ..paths import DATA_DIR


def load_data():
    r"""Loads raw positive and negative examples."""

    # Load the positive examples
    pos_data_1 = pd.read_csv(join(DATA_DIR, 'forta', 'malicious_smart_contracts.csv'))
    pos_data_2 = pd.read_csv(join(DATA_DIR, 'forta', 'phishing_scams.csv'))

    # Find only the contracts
    pos_data_2 = pos_data_2[pos_data_2['is_contract']]
    pos_data_2 = pos_data_2.rename(columns={'address': 'contract_address'})

    pos_data = pd.concat([pos_data_1[['contract_address']], pos_data_2[['contract_address']]])
    pos_data = pos_data.reset_index(drop=True)
    pos_data['label'] = 1

    # Load the negative examples, pick random 10x size
    with jsonlines.open(join(DATA_DIR, 'tintinweb', 'contracts.jsonl')) as reader:
        neg_data = [obj for obj in reader]

    neg_data = pd.DataFrame.from_records(neg_data)
    neg_data = neg_data[neg_data['txcount'] > 1000]
    neg_data = neg_data[['address']]
    neg_data = neg_data.rename(columns={'address': 'contract_address'})
    neg_data['label'] = 0

    data = pd.concat([pos_data, neg_data])
    data = data.reset_index(drop=True)

    return data


def get_w3():
    r"""Returns a Web3 instance."""
    w3 = Web3(HTTPProvider('https://rpc.ankr.com/eth'))
    return w3


def get_contract_code(address, etherscan_api_key, w3):
    r"""Get ABIs and bytecode. If the ABI is a proxy, fetch the underlying contract.
    Arguments:
    --
    address (str): Address of an Ethereum contract address.
    etherscan_api_key (str): API key for Etherscan.
    w3 (web3py instance): Used to interact with the Ethereum blockchain.
    Returns:
    --
    result (Optional[Dict[str,str]]): If something fails, this will return None.
        Otherwise, returns the ABI, bytecode, and opcode.
    """
    abi = get_abi(address, etherscan_api_key)
    if abi is None:
        return None

    contract = w3.eth.contract(w3.toChecksumAddress(address), 
                               abi=json.dumps(abi),
                               )
    fns = contract.all_functions()

    # If we find this definition, then this is likely a proxy contract
    if 'implementation' in fns:
        proxied_address = None

        try:
            # Try directly calling the `implementation` function. This is the
            # most surefire way but it may not work if it is a private function
            #
            # If implementation is a write function, this will definitely fail, good.
            proxied_address = contract.functions.implementation().call()
        except:
            # First try OpenZeppelin proxy
            oz_storage = '0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3'
            result = w3.eth.getStorageAt(w3.toChecksumAddress(address.lower()),
                                         oz_storage,
                                         )
            result = result.hex()
            result = f'0x{result[-40:]}'

            if int(result, base=16) > 0:  # valid address
                proxied_address = result
            else:
                # OZ didn't work... try EIP-1967 
                eip_storage = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
                result = w3.eth.getStorageAt(w3.toChecksumAddress(address.lower()),
                                             eip_storage,
                                             )
                result = result.hex()
                result = f'0x{result[-40:]}'

                if int(result, base=16) > 0:  # valid address
                    proxied_address = result

        if proxied_address is not None:
            # Overwrites the address and ABI
            address = proxied_address
            abi = get_abi(address, etherscan_api_key)

            if abi is None:
                return None

    bytecode = w3.eth.get_code(w3.toChecksumAddress(address))

    if bytecode.hex() == '0x':
        return None

    opcode = bytecode_to_opcode(bytecode)

    result = {
        'abi': json.dumps(abi),  # save as string
        'bytecode': bytecode,
        'opcode': opcode,
    }
    return result


def get_abi(address, etherscan_api_key):
    r"""Fetch the ABI of a deployed contract address.
    Arguments:
    --
    address (str): Address of a deployed contract
    Returns:
    --
    abi_json (optional[str]): None if failed fetch
    Notes:
    --
    Assumes an ethereum contract.
    """
    api_url = 'https://api.etherscan.io/api'
    response = requests.get(f'{api_url}?' +
                            f'module=contract&action=getabi&address={address}' + 
                            f'&apikey={etherscan_api_key}'
                            )

    abi_json = None  # default

    if response.status_code == 200:
        response_json = response.json()

        if response_json['status'] == '0':
            # We expect many contracts to not be verified. If something else happens, print it
            if response_json['result'] != 'Contract source code not verified':
                print(response_json)
            return None

        abi_json = json.loads(response_json['result'])

    return abi_json


def bytecode_to_opcode(bytecode):
    r"""Convert bytecode data to opcodes.
    Argument:
    --
    bytecode (str): Bytecode string.
    Opcode (str): Opcode string.
    """
    opcodes = EvmBytecode(bytecode).disassemble()
    output = []
    for opcode in opcodes:
        output.append(opcode.name)
        if len(opcode.operand) > 0:
            output.append(opcode.operand)
    return ' '.join(output)


def split_data(data, rs=None):
    r"""Split dataset into training and test portions."""

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=rs,
        stratify=data['label'],
    )

    return train_data, test_data
