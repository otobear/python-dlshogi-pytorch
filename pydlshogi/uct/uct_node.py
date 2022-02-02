import shogi

UCT_HASH_SIZE = 2 ** 12
UCT_HASH_LIMIT = UCT_HASH_SIZE * 9 / 10
NOT_EXPANDED = -1

def hash_to_index(hash):
    """hash を index に変換する

    zobrist_hash を UCT_HASH_SIZE 未満の値に変換する

    Args:
        hash (int): hash

    Returns:
        int: index

    """
    return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (UCT_HASH_SIZE - 1)

class NodeHashEntry:
    """ノード衝突情報

    Attributes:
        hash (int): zobrist_hash
        color (int in range(1)): 手番
        moves (int): 手数
        flag (bool): 使用中かどうか

    """
    def __init__(self):
        self.hash = 0
        self.color = 0
        self.moves = 0
        self.flag = False

class NodeHash:
    """ノード衝突情報管理

    Attributes:
        used (int): 使用中のノード数
        enough_size (bool): used が UCT_HASH_LIMIT を超えたかどうか
        node_hash (list of NodeHashEntry): ノード一覧

    """
    def __init__(self):
        self.used = 0
        self.enough_size = True
        self.node_hash = None

    def initialize(self):
        self.used = 0
        self.enough_size = True
        if self.node_hash is None:
            self.node_hash = [NodeHashEntry() for _ in range(UCT_HASH_SIZE)]
        else:
            for i in range(UCT_HASH_SIZE):
                self.node_hash[i].hash = 0
                self.node_hash[i].color = 0
                self.node_hash[i].moves = 0
                self.node_hash[i].flag = False

    def __getitem__(self, i):
        return self.node_hash[i]
    
    def search_empty_index(self, hash, color, moves):
        """未使用のノードを探す

        未使用のノードがあれば、衝突情報を書き込んで該当ノードの index を返す
        全てのノードが使用中の場合は UCT_HASH_SIZE を返す

        Args:
            hash (int): hash
            color (int in range(1)): 手番
            moves (int): 手数

        Returns:
            int: index

        """
        key = hash_to_index(hash)
        i = key

        while True:
            if not self.node_hash[i].flag:
                self.node_hash[i].hash = hash
                self.node_hash[i].color = color
                self.node_hash[i].moves = moves
                self.node_hash[i].flag = True
                self.used += 1
                if self.used > UCT_HASH_LIMIT:
                    self.enough_size = False
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            if i == key:
                return UCT_HASH_SIZE

    def find_same_hash_index(self, hash, color, moves):
        """hash に対応するノードを探す

        対応するノードがあれば、該当ノードの index を返す
        なければ UCT_HASH_SIZE を返す

        Args:
            hash (int): hash
            color (int in range(1)): 手番
            moves (int): 手数

        Returns:
            int: index

        """
        key = hash_to_index(hash)
        i = key

        while True:
            if not self.node_hash[i].flag:
                return UCT_HASH_SIZE
            elif self.node_hash[i] == hash and self.node_hash[i].color == color and self.node_hash[i].moves == moves:
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            if i == key:
                return UCT_HASH_SIZE

    def save_used_hash(self, board, uct_node, index):
        """uct_node[index] と展開済みの子孫を使用中にする

        Args:
            board (shogi.Board): 盤面
            uct_node (list of UctNode): ノード情報
            index (int): 使用中にする index

        Returns:
            int: index

        Todo:
            board unnecessary?

        """
        self.node_hash[index].flag = True
        self.used += 1

        current_node = uct_node[index]
        child_index = current_node.child_index
        child_move = current_node.child_move
        child_num = current_node.child_num
        for i in range(child_num):
            if child_index[i] != NOT_EXPANDED and self.node_hash[child_index[i]].flag == False:
                board.push(child_move[i])
                self.save_used_hash(board, uct_node, child_index[i])
                board.pop()

    def delete_old_hash(self, board, uct_node):
        """盤面と展開済みの子孫以外のノードを未使用にする

        Args:
            board (shogi.Board): 盤面
            uct_node (list of UctNode): ノード情報

        """
        root = self.find_same_hash_index(board.zobrist_hash(), board.turn, board.move_number)
        self.used = 0
        for i in range(UCT_HASH_SIZE):
            self.node_hash[i].flag = False
        if root != UCT_HASH_SIZE:
            self.save_used_hash(board, uct_node, root)

        self.enough_size = True

    def get_usage_rate(self):
        """ノード衝突情報使用率

        Returns:
            ノード衝突情報使用率

        """
        return self.used / UCT_HASH_SIZE

class UctNode:
    """ノード情報

    Attributes:
        move_count (int): 訪問回数
        win (float in [0.0, 1.0]): 合計勝率
        child_num (int): 子ノードの数
        child_move (list of int): 子ノードの指し手
        child_index (list of int): 子ノードの index
        child_move_count (list of int): 子ノードの訪問回数
        child_win (list of float): 子ノードの合計勝率
        nnrate (list of float): policy network の予測確率
        value_win (int): value network の予測勝率
        evaled (bool): 評価済みフラグ

    """
    def __init__(self):
        self.move_count = 0
        self.win = 0.0
        self.child_num = 0
        self.child_move = None
        self.child_index = None
        self.child_move_count = None
        self.child_win = None
        self.nnrate = None
        self.value_win = 0.0
        self.evaled = False
