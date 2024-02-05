from pysui.sui.sui_types import (
    ObjectID,
    SuiU128,
    SuiU64,
    SuiAddress
)
from pysui.sui.sui_txn import SyncTransaction
from pysui import SyncClient

wallet_address1 = "0xc371ebc0f0f5eae548459b9d792135076c924c32c1d613c04ef4f190cb259716"
wallet_address0 = "0xca171941521153181ff729d53489eaae7e99c3f4692884afd7cca61154e4cec4"

# constant
clock = "0x0000000000000000000000000000000000000000000000000000000000000006"
sui = "0x2::sui::SUI"


def suirc20_published_at(env: str) -> str:
    if env == "testnet":
        return "0x6977d5f0299f9ef255d2e791daa7f04dc908bfaf477d01803a915aa4fb755b52"
    elif env == "devnet":
        return "0xc01b650db1d4e122e92f3f1db0b3f66cc175b8cba52c41152f1c2c00d700a306"
    else:
        return "0x0"

def market_published_at(env: str) -> str:
    if env == "testnet":
        return "0x4b128b524bc9d83654ab0705a08ca0ef466547f3d5ee2c9495c08d07a5895d0a"
    elif env == "devnet":
        return "0x8735203320bf379e61fab3d0591689ee26dcbb9f6dd07b58c70dc2fe7e112c8b"
    else:
        return "0x0"

def movetool_published_at(env: str) -> str:
    if env == "testnet":
        return "0x1885b019d728e06158759b66d5aebb936aa4b1b6e2656f3ad355134e2bb49d21"
    else:
        return "0x0"

def set_roles(client: SyncClient ,admin_cap: str, global_config: str, env: str):
    txer = SyncTransaction(
        client=client, initial_sender=SuiAddress(wallet_address0))
    txer.move_call(
        target=market_published_at(env)+"::config::set_roles",
        arguments=[
            ObjectID(admin_cap),  # admin cap
            ObjectID(global_config),  # global config
            SuiAddress(wallet_address1),
            SuiU128(3),  # "00...00011" has token white list and keeper
        ],
        type_arguments=[],
    )

    return txer
