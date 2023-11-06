import numpy as np

class StateNode:
    def __init__(self, state, radius, p_visitation=0):
        self.value = np.array(state)
        self.radius = np.array(radius)
        self.visitation = p_visitation
        self.sub_node = None

    def divide_node(self):
        raise NotImplementedError

    def within(self, target: np.ndarray):
        within = bool(
            ((self.value - self.radius) <= (target + 1e-6)).all() and
            ((self.value + self.radius) >= (target - 1e-6)).all()
        )
        return within

class OneDStateNode(StateNode):
    def divide_node(self):
        sub_node1 = OneDStateNode(self.value - self.radius / 2, self.radius / 2, self.visitation // 2)
        sub_node2 = OneDStateNode(self.value + self.radius / 2, self.radius / 2, self.visitation // 2)
        self.sub_node = [sub_node1, sub_node2]
        return self.sub_node


class TwoDStateNode(StateNode):
    def divide_node(self):
        sub_node1 = TwoDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2], self.radius / 2, self.visitation // 4)
        sub_node2 = TwoDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2], self.radius / 2, self.visitation // 4)
        sub_node3 = TwoDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2], self.radius / 2, self.visitation // 4)
        sub_node4 = TwoDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2], self.radius / 2, self.visitation // 4)
        self.sub_node = [sub_node1, sub_node2, sub_node3, sub_node4]
        return self.sub_node


class ThreeDStateNode(StateNode):
    def divide_node(self):
        sub_node1 = ThreeDStateNode([self.value[0] - self.radius[0] / 2,
                                     self.value[1] - self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node2 = ThreeDStateNode([self.value[0] - self.radius[0] / 2,
                                     self.value[1] - self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node3 = ThreeDStateNode([self.value[0] - self.radius[0] / 2,
                                     self.value[1] + self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node4 = ThreeDStateNode([self.value[0] - self.radius[0] / 2,
                                     self.value[1] + self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node5 = ThreeDStateNode([self.value[0] + self.radius[0] / 2,
                                     self.value[1] - self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node6 = ThreeDStateNode([self.value[0] + self.radius[0] / 2,
                                     self.value[1] - self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node7 = ThreeDStateNode([self.value[0] + self.radius[0] / 2,
                                     self.value[1] + self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        sub_node8 = ThreeDStateNode([self.value[0] + self.radius[0] / 2,
                                     self.value[1] + self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2,],
                                    self.radius / 2, self.visitation // 8)
        self.sub_node = [sub_node1, sub_node2, sub_node3, sub_node4, sub_node5, sub_node6, sub_node7, sub_node8]
        return self.sub_node

class FourDStateNode(StateNode):
    def divide_node(self):
        sub_node1 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                    self.value[2] - self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node2 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                    self.value[2] - self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node3 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                    self.value[2] + self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node4 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                    self.value[2] + self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node5 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                    self.value[2] - self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node6 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                    self.value[2] - self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node7 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                    self.value[2] + self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node8 = FourDStateNode([self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                    self.value[2] + self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node9 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                    self.value[2] - self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                   self.radius / 2, self.visitation // 16)
        sub_node10 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node11 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node12 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node13 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node14 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                     self.value[2] - self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node15 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2, self.value[3] - self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        sub_node16 = FourDStateNode([self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2,
                                     self.value[2] + self.radius[2] / 2, self.value[3] + self.radius[3] / 2],
                                    self.radius / 2, self.visitation // 16)
        self.sub_node =  [sub_node1, sub_node2, sub_node3, sub_node4,
                          sub_node5, sub_node6, sub_node7, sub_node8,
                          sub_node9, sub_node10, sub_node11, sub_node12,
                          sub_node13, sub_node14, sub_node15, sub_node16]
        return self.sub_node


class InformationTree:
    def __init__(self, init_state, init_radius, node_type=TwoDStateNode):
        self.init_state = init_state
        self.init_radius = init_radius
        self.head = node_type(self.init_state, self.init_radius)
        self.data = [self.head]
        self.visitation = [self.data[0].visitation]

    def __len__(self):
        assert len(self.data) == len(self.visitation)
        return len(self.data)

    def divide_node(self, target):
        if not isinstance(target, StateNode):
            head = self.head
            target = self.find_target_node(head, target)
        sub_nodes = target.divide_node()
        idx = self.data.index(target)
        del self.data[idx]
        del self.visitation[idx]
        for sub_node in sub_nodes:
            self.data.append(sub_node)
            self.visitation.append(sub_node.visitation)

    def find_target_node(self, node, target: np.ndarray):
        if node.sub_node is None:
            return node

        for sub_node in node.sub_node:
            if sub_node.within(target):
                return self.find_target_node(sub_node, target)

        return None

    def count_visitation(self, target: np.ndarray):
        node = self.find_target_node(self.head, target)
        idx = self.data.index(node)
        node.visitation += 1
        self.visitation[idx] += 1
