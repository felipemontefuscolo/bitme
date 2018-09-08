from api.position_interface import PositionInterface


def does_reduce_position(qty, position: PositionInterface):
    pos_change = abs(position.signed_qty + qty) - abs(position.signed_qty)
    return pos_change < 0
