class MaskRRWM(object):
    def __init__(self):
        self.mask_ii = None
        self.mask_ij = None
        self.mask_jj = None

    def calculate_mask(self, mask):
        num_graphs, num_nodes = mask.shape
        mask = mask.int()
        # mask_ij[i, j, :ns_gt[i], :ns_gt[j] = 0    mask_ij[i, j, x, y] = x < ns[i] or y < ns[j]
        # mask_1[i,j,x,y] = x<ns[i] mask_2[i,j,x,y] = y<ns[j]
        mask_1 = mask.unsqueeze(1).unsqueeze(3).repeat(1, num_graphs, 1, num_nodes)
        mask_2 = mask.unsqueeze(0).unsqueeze(2).repeat(num_graphs, 1, num_nodes, 1)
        self.mask_ij = (mask_1 | mask_2).reshape(-1, num_nodes, num_nodes).bool()  # (m, m, n, n)
        # mask_ii[i, j, :ns_gt[i], :ns_gt[j] = 0    mask_ii[i, j, x, y] = x < ns[i] or y < ns[i]
        # mask_1[i,j,x,y] = x<ns[i] mask_2[i,j,x,y] = y<ns[i]
        mask_1 = mask.unsqueeze(1).unsqueeze(3).repeat(1, num_graphs, 1, num_nodes)
        mask_2 = mask.unsqueeze(1).unsqueeze(2).repeat(1, num_graphs, num_nodes, 1)
        self.mask_ii = (mask_1 | mask_2).reshape(-1, num_nodes, num_nodes).bool()
        # mask_jj[i, j, :ns_gt[j], :ns_gt[j] = 0    mask_jj[i, j, x, y] = x < ns[j] or y < ns[j]
        # mask_1[i,j,x,y] = x<ns[j] mask_2[i,j,x,y] = y<ns[j]
        mask_1 = mask.unsqueeze(0).unsqueeze(3).repeat(num_graphs, 1, 1, num_nodes)
        mask_2 = mask.unsqueeze(0).unsqueeze(2).repeat(num_graphs, 1, num_nodes, 1)
        self.mask_jj = (mask_1 | mask_2).reshape(-1, num_nodes, num_nodes).bool()

    def set_mask(self, mask_ij, mask_ii, mask_jj):
        self.mask_ij = mask_ij
        self.mask_ii = mask_ii
        self.mask_jj = mask_jj
