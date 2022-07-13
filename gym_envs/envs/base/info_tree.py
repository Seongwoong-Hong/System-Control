import numpy as np

class OneDStateNode:
    def __init__(self, s, r, p_visitation=0):
        self.value = np.array([s])
        self.radius = np.array([r])
        self.visitation = p_visitation

    def divide_node(self):
        sub_node1 = OneDStateNode(self.value[0] - self.radius[0] / 2, self.radius[0] / 2, self.visitation // 2)
        sub_node2 = OneDStateNode(self.value[0] + self.radius[0] / 2, self.radius[0] / 2, self.visitation // 2)
        return [sub_node1, sub_node2]

    def within(self, target: np.ndarray):
        within = bool(
            ((self.value - self.radius) <= target).all() and
            ((self.value + self.radius) >= target).all()
        )
        return within


class TwoDStateNode:
    def __init__(self, s1, s2, r1, r2, p_visitation=0):
        self.value = np.array([s1, s2])
        self.radius = np.array([r1, r2])
        self.visitation = p_visitation

    def divide_node(self):
        sub_node1 = TwoDStateNode(self.value[0] - self.radius[0] / 2, self.value[1] - self.radius[1] / 2, self.radius[0] / 2, self.radius[1] / 2, self.visitation // 4)
        sub_node2 = TwoDStateNode(self.value[0] - self.radius[0] / 2, self.value[1] + self.radius[1] / 2, self.radius[0] / 2, self.radius[1] / 2, self.visitation // 4)
        sub_node3 = TwoDStateNode(self.value[0] + self.radius[0] / 2, self.value[1] - self.radius[1] / 2, self.radius[0] / 2, self.radius[1] / 2, self.visitation // 4)
        sub_node4 = TwoDStateNode(self.value[0] + self.radius[0] / 2, self.value[1] + self.radius[1] / 2, self.radius[0] / 2, self.radius[1] / 2, self.visitation // 4)
        return [sub_node1, sub_node2, sub_node3, sub_node4]

    def within(self, target: np.ndarray):
        within = bool(
            ((self.value - self.radius) <= target).all() and
            ((self.value + self.radius) >= target).all()
        )
        return within


class InformationTree:
    def __init__(self, init_state, init_radius, node_type=TwoDStateNode):
        self.init_state = init_state
        self.init_radius = init_radius
        self.data = [node_type(*self.init_state, *self.init_radius)]
        self.visitation = [self.data[0].visitation]

    def divide_node(self, target: np.ndarray):
        node, _ = self.find_target_node(target)
        sub_nodes = node.divide_node()
        self.data.remove(node)
        self.visitation.remove(node.visitation)
        for node in sub_nodes:
            self.data.append(node)
            self.visitation.append(node.visitation)

    def find_target_node(self, target: np.ndarray):
        idx = 0
        for node in self.data:
            if node.within(target):
                node.visitation += 1
                self.visitation[idx] += 1
                return node, idx
            idx += 1
        raise AssertionError("Target is not in the every nodes")

