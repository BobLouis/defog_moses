#pragma once
// Auto-generated ψ linear model (scaled + Q format)
#define PSI_Q_FRAC 14
#define PSI_NFEAT 21
#define PSI_MIN 0.5 // clamp lower bound
#define PSI_MAX 1.7 // clamp upper bound
static const int32_t PSI_MEAN_Q[PSI_NFEAT] = {9374, 3132, 5436, 9115, 13811, 1, 27, 4096, 574218, 1528335, 1974, 1402, 3902, 10624, 15253, 953, 12103, 15390, 15399, 15356, 26648};
static const int32_t PSI_INV_SCALE_Q[PSI_NFEAT] = {187139, 389451, 157677, 147952, 161445, 16959704, 1529521, 1255470498, 1091, 390, 241632, 426679, 132532, 197651, 256300, 199245, 146300, 230292, 232433, 219755, 136767};
static const int32_t PSI_W_Q[PSI_NFEAT] = {9160, -1671, 1230, 403, -213, -101, -339, 33, 2273, -1816, 434, -1299, 3588, -9237, 884, -494, 1977, -73, -4836, -93, 1098};
static const int32_t PSI_B_Q = 20474;
