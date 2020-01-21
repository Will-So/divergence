def test_all_divergence():
	np.random.seed(948)
	dist_1 = np.random.normal(100, 20, 10_000)
	dist_3 = np.random.normal(100, 20, 10_000)
	dist_2 = np.random.normal(105, 20, 10_001)
	dist_4 = np.random.normal(100,25, 10_000)
	dist_5 = np.random.normal(-200,25, 10_000)

def test_kl_divergence_aligns_with_scipy():
	pass
	