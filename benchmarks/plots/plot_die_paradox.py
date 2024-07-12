import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import parse_output

tail_output = """
Normalizing constant: Z ∈ [0.24996189605243102, 1.0]
Everything from here on is normalized.

Probability masses:
p(0) = 0.0
p(1) ∈ [0.16666666666666666, 1.0]
p(2) ∈ [0.05555555555555555, 1.0]
p(3) ∈ [0.018518518518518517, 1.0]
p(4) ∈ [0.006172839506172839, 1.0]
p(5) ∈ [0.0020576131687242796, 1.0]
p(6) ∈ [0.0006858710562414266, 1.0]
p(7) ∈ [0.00022862368541380886, 0.4930686665563034]
p(8) ∈ [0.00007620789513793628, 0.16580640676776823]
p(9) ∈ [0.0, 0.05575736859676801]
p(10) ∈ [0.0, 0.018750084592269206]
p(11) ∈ [0.0, 0.006305277330423187]
p(12) ∈ [0.0, 0.0021203382852971477]
p(13) ∈ [0.0, 0.0007130272323477806]
p(14) ∈ [0.0, 0.000239776755244641]
p(15) ∈ [0.0, 0.00008063211297882967]
p(16) ∈ [0.0, 0.000027114962152179098]
p(17) ∈ [0.0, 9.118217858276151e-6]
p(18) ∈ [0.0, 3.066273758538306e-6]
p(19) ∈ [0.0, 1.0311263569741178e-6]
p(20) ∈ [0.0, 3.467471099363137e-7]
p(21) ∈ [0.0, 1.1660409748618615e-7]
p(22) ∈ [0.0, 3.9211618960761485e-8]
p(23) ∈ [0.0, 1.318608088970544e-8]
p(24) ∈ [0.0, 4.4342144965717184e-9]
p(25) ∈ [0.0, 1.4911373869211873e-9]
p(26) ∈ [0.0, 5.014395917006774e-10]
p(27) ∈ [0.0, 1.6862407604446422e-10]
p(28) ∈ [0.0, 5.6704894253388586e-11]
p(29) ∈ [0.0, 1.9068718463667703e-11]
p(30) ∈ [0.0, 6.412427509727566e-12]
p(31) ∈ [0.0, 2.156370741214559e-12]
p(32) ∈ [0.0, 7.251442244785363e-13]
p(33) ∈ [0.0, 2.438514566370001e-13]
p(34) ∈ [0.0, 8.200235332046943e-14]
p(35) ∈ [0.0, 2.7575746492690008e-14]
p(36) ∈ [0.0, 9.273170388870884e-15]
p(37) ∈ [0.0, 3.1183811863017792e-15]
p(38) ∈ [0.0, 1.0486490396802618e-15]
p(39) ∈ [0.0, 3.5263963663354266e-16]
p(40) ∈ [0.0, 1.185856360131254e-16]
p(41) ∈ [0.0, 3.9877970618631953e-17]
p(42) ∈ [0.0, 1.3410161585543629e-17]
p(43) ∈ [0.0, 4.509568339627793e-18]
p(44) ∈ [0.0, 1.5164773727779782e-18]
p(45) ∈ [0.0, 5.099609206360116e-19]
p(46) ∈ [0.0, 1.7148962803153073e-19]
p(47) ∈ [0.0, 5.766852190500191e-20]
p(48) ∈ [0.0, 1.9392767112983748e-20]
p(49) ∈ [0.0, 6.521398570227696e-21]
p(50) ∈ [0.0, 2.193015522952074e-21]
p(51) ∈ [0.0, 7.3746712949962205e-22]
p(52) ∈ [0.0, 2.479954024038606e-22]
p(53) ∈ [0.0, 8.339587915624415e-23]
p(54) ∈ [0.0, 2.8044361277782343e-23]
p(55) ∈ [0.0, 9.430756140903284e-24]
p(56) ∈ [0.0, 3.171374113613545e-24]
p(57) ∈ [0.0, 1.0664694981217882e-24]
p(58) ∈ [0.0, 3.5863229933733834e-25]
p(59) ∈ [0.0, 1.206008482703914e-25]
p(60) ∈ [0.0, 4.055564607653198e-26]
p(61) ∈ [0.0, 1.3638050248181608e-26]
p(62) ∈ [0.0, 4.5862027255325984e-27]
p(63) ∈ [0.0, 1.5422479795077045e-27]
p(64) ∈ [0.0, 5.1862705873285995e-28]
p(65) ∈ [0.0, 1.7440387643481017e-28]
p(66) ∈ [0.0, 5.864852518456024e-29]
p(67) ∈ [0.0, 1.9722322557490232e-29]
p(68) ∈ [0.0, 6.632221455486626e-30]
p(69) ∈ [0.0, 2.2302830362092315e-30]
p(70) ∈ [0.0, 7.499994466390597e-31]
p(71) ∈ [0.0, 2.5220976926541333e-31]
p(72) ∈ [0.0, 8.481308619354955e-32]
p(73) ∈ [0.0, 2.852093957591558e-32]
p(74) ∈ [0.0, 9.591019862626978e-33]
p(75) ∈ [0.0, 3.22526758841367e-33]
p(76) ∈ [0.0, 1.084592792619087e-33]
p(77) ∈ [0.0, 3.647267997319401e-34]
p(78) ∈ [0.0, 1.2265030650025888e-34]
p(79) ∈ [0.0, 4.1244837768059633e-35]
p(80) ∈ [0.0, 1.386981158917828e-35]
p(81) ∈ [0.0, 4.664139415485309e-36]
p(82) ∈ [0.0, 1.5684565249651295e-36]
p(83) ∈ [0.0, 5.274404668389867e-37]
p(84) ∈ [0.0, 1.7736764878804217e-37]
p(85) ∈ [0.0, 5.964518237506025e-38]
p(86) ∈ [0.0, 2.005747837817672e-38]
p(87) ∈ [0.0, 6.744927634913954e-39]
p(88) ∈ [0.0, 2.2681838635172184e-39]
p(89) ∈ [0.0, 7.627447345898068e-40]
p(90) ∈ [0.0, 2.564957539387142e-40]
p(91) ∈ [0.0, 8.62543768643259e-41]
p(92) ∈ [0.0, 2.900561671687864e-41]
p(93) ∈ [0.0, 9.754007062735331e-42]
p(94) ∈ [0.0, 3.2800769143628476e-42]
p(95) ∈ [0.0, 1.1030240694862652e-42]
p(96) ∈ [0.0, 3.709248684195496e-43]
p(97) ∈ [0.0, 1.2473459266953316e-43]
p(98) ∈ [0.0, 4.194574139697757e-44]
p(99) ∈ [0.0, 1.4105511419783297e-44]
p(100) ∈ [0.0, 4.7434005404890416e-45]
p(101) ∈ [0.0, 1.595110451362663e-45]
p(102) ∈ [0.0, 5.364036476211379e-46]
p(103) ∈ [0.0, 1.8038178668785114e-46]
p(104) ∈ [0.0, 6.065877648856472e-47]
p(105) ∈ [0.0, 2.0398329746322822e-47]
p(106) ∈ [0.0, 6.859549112701925e-48]
p(107) ∈ [0.0, 2.306728767243897e-48]
p(108) ∈ [0.0, 7.757066125202867e-49]
p(109) ∈ [0.0, 2.608545734775053e-49]
p(110) ∈ [0.0, 8.772016043933319e-50]
p(111) ∈ [0.0, 2.949853033021832e-50]
p(112) ∈ [0.0, 9.919764023284142e-51]
p(113) ∈ [0.0, 3.335817655188048e-51]
p(114) ∈ [0.0, 1.1217685624925018e-51]
p(115) ∈ [0.0, 3.772282654117546e-52]
p(116) ∈ [0.0, 1.2685429863480626e-52]
p(117) ∈ [0.0, 4.265855599278531e-53]
p(118) ∈ [0.0, 1.434521666962491e-53]
p(119) ∈ [0.0, 4.824008607635198e-54]
p(120) ∈ [0.0, 1.6222173273836624e-54]
p(121) ∈ [0.0, 5.455191462757023e-55]
p(122) ∈ [0.0, 1.8344714603272716e-55]
p(123) ∈ [0.0, 6.168959534656692e-56]
p(124) ∈ [0.0, 2.0744973450523166e-56]
p(125) ∈ [0.0, 6.976118436913374e-57]
p(126) ∈ [0.0, 2.3459286926499004e-57]
p(127) ∈ [0.0, 7.888887611020944e-58]
p(128) ∈ [0.0, 2.6528746561781122e-58]
p(129) ∈ [0.0, 8.921085314436803e-59]
p(130) ∈ [0.0, 2.9999820384320733e-59]
p(131) ∈ [0.0, 1.0088337812833965e-59]
p(132) ∈ [0.0, 3.3925056391020126e-60]
p(133) ∈ [0.0, 1.1408315943482346e-60]
p(134) ∈ [0.0, 3.836387806293043e-61]
p(135) ∈ [0.0, 1.290100263105211e-61]
p(136) ∈ [0.0, 4.3383483967235885e-62]
p(137) ∈ [0.0, 1.4588995405715382e-62]
p(138) ∈ [0.0, 4.905986506494611e-63]
p(139) ∈ [0.0, 1.6497848503316442e-63]
p(140) ∈ [0.0, 5.547895512514483e-64]
p(141) ∈ [0.0, 1.865645972660679e-64]
p(142) ∈ [0.0, 6.273793166172081e-65]
p(143) ∈ [0.0, 2.10975079241716e-65]
p(144) ∈ [0.0, 7.094668708724128e-66]
p(145) ∈ [0.0, 2.385794770996664e-66]
p(146) ∈ [0.0, 8.022949235551622e-67]
p(147) ∈ [0.0, 2.6979568912940812e-67]
p(148) ∈ [0.0, 9.072687827845584e-68]
p(149) ∈ [0.0, 3.050962922615694e-68]
p(150) ∈ [0.0, 1.0259776299815752e-68]
p(151) ∈ [0.0, 3.450156963297851e-69]
p(152) ∈ [0.0, 1.1602185782166048e-69]
p(153) ∈ [0.0, 3.901582344103777e-70]
p(154) ∈ [0.0, 1.3120238783989213e-70]
p(155) ∈ [0.0, 4.412073117181301e-71]
p(156) ∈ [0.0, 1.4836916851779398e-71]
p(157) ∈ [0.0, 4.989357515617294e-72]
p(158) ∈ [0.0, 1.6778208483160214e-72]
p(159) ∈ [0.0, 5.6421749498454315e-73]
p(160) ∈ [0.0, 1.897350256233511e-73]
p(161) ∈ [0.0, 6.380408312096013e-74]
p(162) ∈ [0.0, 2.1456033273412475e-74]
p(163) ∈ [0.0, 7.215233591822447e-75]
p(164) ∈ [0.0, 2.4263383227072734e-75]
p(165) ∈ [0.0, 8.159289067106911e-76]
p(166) ∈ [0.0, 2.7438052417326565e-76]
p(167) ∈ [0.0, 9.226866633405127e-77]
p(168) ∈ [0.0, 3.1028101621703963e-77]
p(169) ∈ [0.0, 1.0434128166123635e-77]
p(170) ∈ [0.0, 3.508787998520025e-78]
p(171) ∈ [0.0, 1.179935019250585e-78]
p(172) ∈ [0.0, 3.967884780274885e-79]
p(173) ∈ [0.0, 1.334320057687301e-79]
p(174) ∈ [0.0, 4.487050695618486e-80]
p(175) ∈ [0.0, 1.5089051407910908e-80]
p(176) ∈ [0.0, 5.074145309142653e-81]
p(177) ∈ [0.0, 1.7063332824749835e-81]
p(178) ∈ [0.0, 5.738056546460634e-82]
p(179) ∈ [0.0, 1.9295933138350688e-82]
p(180) ∈ [0.0, 6.488835247002991e-83]
p(181) ∈ [0.0, 2.182065130556691e-83]
p(182) ∈ [0.0, 7.337847321968219e-84]
p(183) ∈ [0.0, 2.467570860581023e-84]
p(184) ∈ [0.0, 8.297945820921432e-85]
p(185) ∈ [0.0, 2.7904327266506304e-85]
p(186) ∈ [0.0, 9.383665511928143e-86]
p(187) ∈ [0.0, 3.155538479705272e-86]
p(188) ∈ [0.0, 1.0611442920938708e-86]
p(189) ∈ [0.0, 3.568415393713004e-87]
p(190) ∈ [0.0, 1.1999865161562303e-87]
p(191) ∈ [0.0, 4.0353139421317566e-88]
p(192) ∈ [0.0, 1.356995132222211e-88]
p(193) ∈ [0.0, 4.563302422765131e-89]
p(194) ∈ [0.0, 1.5345470670564045e-89]
p(195) ∈ [0.0, 5.160373963522019e-90]
p(196) ∈ [0.0, 1.735330249236151e-90]
p(197) ∈ [0.0, 5.835567529022072e-91]
p(198) ∈ [0.0, 1.9623843012456228e-91]
p(199) ∈ [0.0, 6.599104759945459e-92]

Asymptotics: p(n) <= 1013.8964742397486 * 0.33627994046612986^n for n >= 9

Moments:
0-th (raw) moment ∈ [0.24996189605243102, 513.7556612105643]
1-th (raw) moment ∈ [0.37463801249809475, 774.0481016061832]
2-th (raw) moment ∈ [0.7465325407712239, 779.1928120488355]
3-th (raw) moment ∈ [2.0289590001524154, 719.8679447353212]
4-th (raw) moment ∈ [7.171772595640908, 659.7204562198358]
Total time: 0.15435s
"""

geom_bound_output = """
Probability masses:
p(0) = 0.0
p(1) ∈ [0.663840786787213, 0.6666779569164071]
p(2) ∈ [0.22128026226240435, 0.22222598563880233]
p(3) ∈ [0.07376008742080145, 0.07558553235580284]
p(4) ∈ [0.024586695806933817, 0.025660659917368406]
p(5) ∈ [0.008195565268977939, 0.008852187428527879]
p(6) ∈ [0.002731855089659313, 0.0031423203323852145]
p(7) ∈ [0.0009106183632197711, 0.0011703570218759183]
p(8) ∈ [0.00030353945440659034, 0.0004689773674583643]
p(9) ∈ [0.00010117981813553011, 0.00020691802185250264]
p(10) ∈ [0.000033726606045176705, 0.0001014305389652899]
p(11) ∈ [0.0, 0.00006507360055879575]
p(12) ∈ [0.0, 0.00004174850624755919]
p(13) ∈ [0.0, 0.000026784099218971258]
p(14) ∈ [0.0, 0.000017183560214531954]
p(15) ∈ [0.0, 0.000011024255071355972]
p(16) ∈ [0.0, 7.07270195238922e-6]
p(17) ∈ [0.0, 4.5375503908018245e-6]
p(18) ∈ [0.0, 2.91110295438231e-6]
p(19) ∈ [0.0, 1.8676421595652827e-6]
p(20) ∈ [0.0, 1.198201262835718e-6]
p(21) ∈ [0.0, 7.687159228592716e-7]
p(22) ∈ [0.0, 4.931760534610634e-7]
p(23) ∈ [0.0, 3.164011730142817e-7]
p(24) ∈ [0.0, 2.02989787485124e-7]
p(25) ∈ [0.0, 1.3022977579604583e-7]
p(26) ∈ [0.0, 8.354998896252971e-8]
p(27) ∈ [0.0, 5.360218592844103e-8]
p(28) ∈ [0.0, 3.438892538448717e-8]
p(29) ∈ [0.0, 2.2062499292073573e-8]
p(30) ∈ [0.0, 1.41543787591665e-8]
p(31) ∈ [0.0, 9.080858673610136e-9]
p(32) ∈ [0.0, 5.825899931968214e-9]
p(33) ∈ [0.0, 3.737654250246557e-9]
p(34) ∈ [0.0, 2.397922974565498e-9]
p(35) ∈ [0.0, 1.5384073022724724e-9]
p(36) ∈ [0.0, 9.86977919136085e-10]
p(37) ∈ [0.0, 6.33203841026533e-10]
p(38) ∈ [0.0, 4.06237157404402e-10]
p(39) ∈ [0.0, 2.60624805731546e-10]
p(40) ∈ [0.0, 1.6720599808399025e-10]
p(41) ∈ [0.0, 1.0727238996606124e-10]
p(42) ∈ [0.0, 6.882148834906257e-11]
p(43) ∈ [0.0, 4.4152994634301093e-11]
p(44) ∈ [0.0, 2.8326718615686188e-11]
p(45) ∈ [0.0, 1.8173240437668988e-11]
p(46) ∈ [0.0, 1.1659192597847851e-11]
p(47) ∈ [0.0, 7.480051370032179e-12]
p(48) ∈ [0.0, 4.798888776281833e-12]
p(49) ∈ [0.0, 3.078766755451397e-12]
p(50) ∈ [0.0, 1.9752082568200042e-12]
p(51) ∈ [0.0, 1.2672111815232017e-12]
p(52) ∈ [0.0, 8.129898065344934e-13]
p(53) ∈ [0.0, 5.215803294400551e-13]
p(54) ∈ [0.0, 3.3462417101936206e-13]
p(55) ∈ [0.0, 2.146809024615724e-13]
p(56) ∈ [0.0, 1.3773030723189573e-13]
p(57) ∈ [0.0, 8.836201689429703e-14]
p(58) ∈ [0.0, 5.6689382217684356e-14]
p(59) ∈ [0.0, 3.636954167837835e-14]
p(60) ∈ [0.0, 2.3333180044475194e-14]
p(61) ∈ [0.0, 1.4969594497572756e-14]
p(62) ∈ [0.0, 9.603867068039017e-15]
p(63) ∈ [0.0, 6.1614402898835824e-15]
p(64) ∈ [0.0, 3.952922939983206e-15]
p(65) ∈ [0.0, 2.536030381581562e-15]
p(66) ∈ [0.0, 1.6270112506498916e-15]
p(67) ∈ [0.0, 1.04382251449624e-15]
p(68) ∈ [0.0, 6.696729609792422e-16]
p(69) ∈ [0.0, 4.296342227137515e-16]
p(70) ∈ [0.0, 2.7563538634878675e-16]
p(71) ∈ [0.0, 1.768361601358373e-16]
p(72) ∈ [0.0, 1.134507000201251e-16]
p(73) ∈ [0.0, 7.278523422567796e-17]
p(74) ∈ [0.0, 4.6695968560326566e-17]
p(75) ∈ [0.0, 2.995818455465988e-17]
p(76) ∈ [0.0, 1.92199208942757e-17]
p(77) ∈ [0.0, 1.2330699095207893e-17]
p(78) ∈ [0.0, 7.910861913164527e-18]
p(79) ∈ [0.0, 5.075278840717021e-18]
p(80) ∈ [0.0, 3.256087085550698e-18]
p(81) ∈ [0.0, 2.088969580081674e-18]
p(82) ∈ [0.0, 1.340195698656678e-18]
p(83) ∈ [0.0, 8.598136266913169e-19]
p(84) ∈ [0.0, 5.516205382430934e-19]
p(85) ∈ [0.0, 3.5389671524808477e-19]
p(86) ∈ [0.0, 2.27045362491907e-19]
p(87) ∈ [0.0, 1.456628287520124e-19]
p(88) ∈ [0.0, 9.345119163486276e-20]
p(89) ∈ [0.0, 5.995438433262744e-20]
p(90) ∈ [0.0, 3.8464230769246095e-20]
p(91) ∈ [0.0, 2.4677045142546306e-20]
p(92) ∈ [0.0, 1.5831762257784622e-20]
p(93) ∈ [0.0, 1.0156997920098255e-20]
p(94) ∈ [0.0, 6.516305959442595e-21]
p(95) ∈ [0.0, 4.180589943121335e-21]
p(96) ∈ [0.0, 2.6820920290277865e-21]
p(97) ∈ [0.0, 1.7207183077141145e-21]
p(98) ∈ [0.0, 1.1039410514097056e-21]
p(99) ∈ [0.0, 7.08242505193385e-22]
p(100) ∈ [0.0, 4.543788325672477e-22]
p(101) ∈ [0.0, 2.9151049530527843e-22]
p(102) ∈ [0.0, 1.8702096748873537e-22]
p(103) ∈ [0.0, 1.1998484735101503e-22]
p(104) ∈ [0.0, 7.697727044810362e-23]
p(105) ∈ [0.0, 4.9385404044441274e-23]
p(106) ∈ [0.0, 3.168361411667593e-23]
p(107) ∈ [0.0, 2.0326884489819573e-23]
p(108) ∈ [0.0, 1.3040880738570755e-23]
p(109) ∈ [0.0, 8.366484815850658e-24]
p(110) ∈ [0.0, 5.367587479488844e-24]
p(111) ∈ [0.0, 3.443620108576754e-24]
p(112) ∈ [0.0, 2.2092829408946047e-24]
p(113) ∈ [0.0, 1.4173837296313152e-24]
p(114) ∈ [0.0, 9.09334245893413e-25]
p(115) ∈ [0.0, 5.833909007616657e-25]
p(116) ∈ [0.0, 3.742792538920842e-25]
p(117) ∈ [0.0, 2.401219486130527e-25]
p(118) ∈ [0.0, 1.5405222065114567e-25]
p(119) ∈ [0.0, 9.883347534295009e-26]
p(120) ∈ [0.0, 6.34074329281204e-26]
p(121) ∈ [0.0, 4.0679562633844723e-26]
p(122) ∈ [0.0, 2.6098309609172037e-26]
p(123) ∈ [0.0, 1.6743586222569646e-26]
p(124) ∈ [0.0, 1.0741986120591435e-26]
p(125) ∈ [0.0, 6.89160997417852e-27]
p(126) ∈ [0.0, 4.42136933552302e-27]
p(127) ∈ [0.0, 2.8365660381750572e-27]
p(128) ∈ [0.0, 1.819822384952678e-27]
p(129) ∈ [0.0, 1.1675220912203808e-27]
p(130) ∈ [0.0, 7.490334467575289e-28]
p(131) ∈ [0.0, 4.805485982496585e-28]
p(132) ∈ [0.0, 3.082999247621923e-28]
p(133) ∈ [0.0, 1.977923647151976e-28]
p(134) ∈ [0.0, 1.2689532626323678e-28]
p(135) ∈ [0.0, 8.141074530677301e-29]
p(136) ∈ [0.0, 5.222973648103853e-29]
p(137) ∈ [0.0, 3.350841909872276e-29]
p(138) ∈ [0.0, 2.1497603207385014e-29]
p(139) ∈ [0.0, 1.3791965007378883e-29]
p(140) ∈ [0.0, 8.848349135936159e-30]
p(141) ∈ [0.0, 5.6767315164687716e-30]
p(142) ∈ [0.0, 3.6419540204615126e-30]
p(143) ∈ [0.0, 2.336525701220159e-30]
p(144) ∈ [0.0, 1.499017374132181e-30]
p(145) ∈ [0.0, 9.617069851946007e-31]
p(146) ∈ [0.0, 6.169910645015201e-31]
p(147) ∈ [0.0, 3.958357166322225e-31]
p(148) ∈ [0.0, 2.539516754401216e-31]
p(149) ∈ [0.0, 1.629247961945913e-31]
p(150) ∈ [0.0, 1.045257495113788e-31]
p(151) ∈ [0.0, 6.705935846540103e-32]
p(152) ∈ [0.0, 4.3022485644091067e-32]
p(153) ∈ [0.0, 2.7601431229781344e-32]
p(154) ∈ [0.0, 1.7707926321011714e-32]
p(155) ∈ [0.0, 1.1360666480658567e-32]
p(156) ∈ [0.0, 7.28852947234226e-33]
p(157) ∈ [0.0, 4.6760163199617364e-33]
p(158) ∈ [0.0, 2.9999369156041668e-33]
p(159) ∈ [0.0, 1.9246343215667574e-33]
p(160) ∈ [0.0, 1.234765055386749e-33]
p(161) ∈ [0.0, 7.921737261565083e-34]
p(162) ∈ [0.0, 5.0822560103640995e-34]
p(163) ∈ [0.0, 3.260563346401489e-34]
p(164) ∈ [0.0, 2.091841362225127e-34]
p(165) ∈ [0.0, 1.3420381142250198e-34]
p(166) ∈ [0.0, 8.609956436260649e-35]
p(167) ∈ [0.0, 5.523788709765106e-35]
p(168) ∈ [0.0, 3.5438322988054604e-35]
p(169) ∈ [0.0, 2.2735748997522472e-35]
p(170) ∈ [0.0, 1.458630767185522e-35]
p(171) ∈ [0.0, 9.357966237277122e-36]
p(172) ∈ [0.0, 6.003680579629539e-36]
p(173) ∈ [0.0, 3.8517108940444965e-36]
p(174) ∈ [0.0, 2.4710969570297324e-36]
p(175) ∈ [0.0, 1.585352675478104e-36]
p(176) ∈ [0.0, 1.0170961113021768e-36]
p(177) ∈ [0.0, 6.525264161263262e-37]
p(178) ∈ [0.0, 4.186337151535584e-37]
p(179) ∈ [0.0, 2.6857791980844844e-37]
p(180) ∈ [0.0, 1.7230838414955173e-37]
p(181) ∈ [0.0, 1.10545867915742e-37]
p(182) ∈ [0.0, 7.092161518175591e-38]
p(183) ∈ [0.0, 4.5500348360581316e-38]
p(184) ∈ [0.0, 2.9191124534157824e-38]
p(185) ∈ [0.0, 1.8727807198657325e-38]
p(186) ∈ [0.0, 1.201497948664758e-38]
p(187) ∈ [0.0, 7.708309388987723e-39]
p(188) ∈ [0.0, 4.9453295948102454e-39]
p(189) ∈ [0.0, 3.172717072857105e-39]
p(190) ∈ [0.0, 2.035482859416006e-39]
p(191) ∈ [0.0, 1.3058808509658002e-39]
p(192) ∈ [0.0, 8.37798652555803e-40]
p(193) ∈ [0.0, 5.3749664964089546e-40]
p(194) ∈ [0.0, 3.448354177866676e-40]
p(195) ∈ [0.0, 2.2123201221728734e-40]
p(196) ∈ [0.0, 1.4193322583815601e-40]
p(197) ∈ [0.0, 9.105843406169969e-41]
p(198) ∈ [0.0, 5.84192909363148e-41]
p(199) ∈ [0.0, 3.747937891386675e-41]

Asymptotics: p(n) <= 0.008586332324872346 * 0.6415582646274244^n for n >= 11

Moments:
0-th (raw) moment ∈ [0.9957443168777971, 1.004273871364435]
1-th (raw) moment ∈ [1.4934478422864697, 1.5207868542875842]
2-th (raw) moment ∈ [2.9850407212404546, 1.5628598968044758]
3-th (raw) moment ∈ [8.188921127587038, 1.5375707183128753]
4-th (raw) moment ∈ [29.566597822534384, 1.650977563607712]
Total time: 0.17380s
"""

rest_output = """
Normalizing constant: Z ∈ [0.24999999999999997, 0.25000000000000006]
Everything from here on is normalized.

Probability masses:
p(0) = 0
p(1) ∈ [0.6666666666666666, 0.6666666666666667]
p(2) ∈ [0.2222222222222222, 0.22222222222222224]
p(3) ∈ [0.07407407407407407, 0.07407407407407408]
p(4) ∈ [0.024691358024691357, 0.02469135802469136]
p(5) ∈ [0.008230452674897118, 0.008230452674897122]
p(6) ∈ [0.0027434842249657062, 0.0027434842249657075]
p(7) ∈ [0.0009144947416552355, 0.0009144947416552365]
p(8) ∈ [0.0003048315805517451, 0.0003048315805517462]
p(9) ∈ [0.00010161052685058171, 0.00010161052685058272]
p(10) ∈ [0.00003387017561686057, 0.00003387017561686156]
p(11) ∈ [0.000011290058538953524, 0.000011290058538954512]
p(12) ∈ [3.763352846317841e-6, 3.763352846318829e-6]
p(13) ∈ [1.2544509487726136e-6, 1.2544509487736009e-6]
p(14) ∈ [4.1815031625753794e-7, 4.18150316258525e-7]
p(15) ∈ [1.3938343875251265e-7, 1.393834387534997e-7]
p(16) ∈ [4.6461146250837546e-8, 4.6461146251824584e-8]
p(17) ∈ [1.548704875027918e-8, 1.5487048751266215e-8]
p(18) ∈ [5.162349583426394e-9, 5.1623495844134266e-9]
p(19) ∈ [1.7207831944754647e-9, 1.7207831954624965e-9]
p(20) ∈ [5.735943981584882e-10, 5.735943991455199e-10]
p(21) ∈ [1.9119813271949607e-10, 1.911981337065277e-10]
p(22) ∈ [6.373271090649869e-11, 6.37327118935303e-11]
p(23) ∈ [2.1244236968832897e-11, 2.1244237955864502e-11]
p(24) ∈ [7.0814123229442994e-12, 7.0814133099759005e-12]
p(25) ∈ [2.3604707743147665e-12, 2.3604717613463675e-12]
p(26) ∈ [7.868235914382555e-13, 7.868245784698564e-13]
p(27) ∈ [2.622745304794185e-13, 2.622755175110193e-13]
p(28) ∈ [8.74248434931395e-14, 8.742583052474031e-14]
p(29) ∈ [2.914161449771316e-14, 2.914260152931397e-14]
p(30) ∈ [9.713871499237721e-15, 9.71485853083852e-15]
p(31) ∈ [3.2379571664125738e-15, 3.2389441980133707e-15]
p(32) ∈ [1.079319055470858e-15, 1.0803060870716546e-15]
p(33) ∈ [3.5977301849028596e-16, 3.6076005009108243e-16]
p(34) ∈ [1.1992433949676198e-16, 1.209113710975584e-16]
p(35) ∈ [3.9974779832254e-17, 4.09618114330504e-17]
p(36) ∈ [1.3324926610751333e-17, 1.431195821154773e-17]
p(37) ∈ [4.441642203583777e-18, 5.428673804380173e-18]
p(38) ∈ [1.4805474011945926e-18, 2.467579001990988e-18]
p(39) ∈ [4.935158003981975e-19, 1.4805474011945927e-18]
p(40) ∈ [0.0, 9.870316007963952e-19]
p(41) ∈ [0.0, 9.870316007963952e-19]
p(42) ∈ [0.0, 9.870316007963952e-19]
p(43) ∈ [0.0, 9.870316007963952e-19]
p(44) ∈ [0.0, 9.870316007963952e-19]
p(45) ∈ [0.0, 9.870316007963952e-19]
p(46) ∈ [0.0, 9.870316007963952e-19]
p(47) ∈ [0.0, 9.870316007963952e-19]
p(48) ∈ [0.0, 9.870316007963952e-19]
p(49) ∈ [0.0, 9.870316007963952e-19]
p(50) ∈ [0.0, 9.870316007963952e-19]
p(51) ∈ [0.0, 9.870316007963952e-19]
p(52) ∈ [0.0, 9.870316007963952e-19]
p(53) ∈ [0.0, 9.870316007963952e-19]
p(54) ∈ [0.0, 9.870316007963952e-19]
p(55) ∈ [0.0, 9.870316007963952e-19]
p(56) ∈ [0.0, 9.870316007963952e-19]
p(57) ∈ [0.0, 9.870316007963952e-19]
p(58) ∈ [0.0, 9.870316007963952e-19]
p(59) ∈ [0.0, 9.870316007963952e-19]
p(60) ∈ [0.0, 9.870316007963952e-19]
p(61) ∈ [0.0, 9.870316007963952e-19]
p(62) ∈ [0.0, 9.870316007963952e-19]
p(63) ∈ [0.0, 9.870316007963952e-19]
p(64) ∈ [0.0, 9.870316007963952e-19]
p(65) ∈ [0.0, 9.870316007963952e-19]
p(66) ∈ [0.0, 9.870316007963952e-19]
p(67) ∈ [0.0, 9.870316007963952e-19]
p(68) ∈ [0.0, 9.870316007963952e-19]
p(69) ∈ [0.0, 9.870316007963952e-19]
p(70) ∈ [0.0, 9.870316007963952e-19]
p(71) ∈ [0.0, 9.870316007963952e-19]
p(72) ∈ [0.0, 9.870316007963952e-19]
p(73) ∈ [0.0, 9.870316007963952e-19]
p(74) ∈ [0.0, 9.870316007963952e-19]
p(75) ∈ [0.0, 9.870316007963952e-19]
p(76) ∈ [0.0, 9.870316007963952e-19]
p(77) ∈ [0.0, 9.870316007963952e-19]
p(78) ∈ [0.0, 9.870316007963952e-19]
p(79) ∈ [0.0, 9.870316007963952e-19]
p(80) ∈ [0.0, 9.870316007963952e-19]
p(81) ∈ [0.0, 9.870316007963952e-19]
p(82) ∈ [0.0, 9.870316007963952e-19]
p(83) ∈ [0.0, 9.870316007963952e-19]
p(84) ∈ [0.0, 9.870316007963952e-19]
p(85) ∈ [0.0, 9.870316007963952e-19]
p(86) ∈ [0.0, 9.870316007963952e-19]
p(87) ∈ [0.0, 9.870316007963952e-19]
p(88) ∈ [0.0, 9.870316007963952e-19]
p(89) ∈ [0.0, 9.870316007963952e-19]
p(90) ∈ [0.0, 9.870316007963952e-19]
p(91) ∈ [0.0, 9.870316007963952e-19]
p(92) ∈ [0.0, 9.870316007963952e-19]
p(93) ∈ [0.0, 9.870316007963952e-19]
p(94) ∈ [0.0, 9.870316007963952e-19]
p(95) ∈ [0.0, 9.870316007963952e-19]
p(96) ∈ [0.0, 9.870316007963952e-19]
p(97) ∈ [0.0, 9.870316007963952e-19]
p(98) ∈ [0.0, 9.870316007963952e-19]
p(99) ∈ [0.0, 9.870316007963952e-19]
p(100) ∈ [0.0, 9.870316007963952e-19]
p(101) ∈ [0.0, 9.870316007963952e-19]
p(102) ∈ [0.0, 9.870316007963952e-19]
p(103) ∈ [0.0, 9.870316007963952e-19]
p(104) ∈ [0.0, 9.870316007963952e-19]
p(105) ∈ [0.0, 9.870316007963952e-19]
p(106) ∈ [0.0, 9.870316007963952e-19]
p(107) ∈ [0.0, 9.870316007963952e-19]
p(108) ∈ [0.0, 9.870316007963952e-19]
p(109) ∈ [0.0, 9.870316007963952e-19]
p(110) ∈ [0.0, 9.870316007963952e-19]
p(111) ∈ [0.0, 9.870316007963952e-19]
p(112) ∈ [0.0, 9.870316007963952e-19]
p(113) ∈ [0.0, 9.870316007963952e-19]
p(114) ∈ [0.0, 9.870316007963952e-19]
p(115) ∈ [0.0, 9.870316007963952e-19]
p(116) ∈ [0.0, 9.870316007963952e-19]
p(117) ∈ [0.0, 9.870316007963952e-19]
p(118) ∈ [0.0, 9.870316007963952e-19]
p(119) ∈ [0.0, 9.870316007963952e-19]
p(120) ∈ [0.0, 9.870316007963952e-19]
p(121) ∈ [0.0, 9.870316007963952e-19]
p(122) ∈ [0.0, 9.870316007963952e-19]
p(123) ∈ [0.0, 9.870316007963952e-19]
p(124) ∈ [0.0, 9.870316007963952e-19]
p(125) ∈ [0.0, 9.870316007963952e-19]
p(126) ∈ [0.0, 9.870316007963952e-19]
p(127) ∈ [0.0, 9.870316007963952e-19]
p(128) ∈ [0.0, 9.870316007963952e-19]
p(129) ∈ [0.0, 9.870316007963952e-19]
p(130) ∈ [0.0, 9.870316007963952e-19]
p(131) ∈ [0.0, 9.870316007963952e-19]
p(132) ∈ [0.0, 9.870316007963952e-19]
p(133) ∈ [0.0, 9.870316007963952e-19]
p(134) ∈ [0.0, 9.870316007963952e-19]
p(135) ∈ [0.0, 9.870316007963952e-19]
p(136) ∈ [0.0, 9.870316007963952e-19]
p(137) ∈ [0.0, 9.870316007963952e-19]
p(138) ∈ [0.0, 9.870316007963952e-19]
p(139) ∈ [0.0, 9.870316007963952e-19]
p(140) ∈ [0.0, 9.870316007963952e-19]
p(141) ∈ [0.0, 9.870316007963952e-19]
p(142) ∈ [0.0, 9.870316007963952e-19]
p(143) ∈ [0.0, 9.870316007963952e-19]
p(144) ∈ [0.0, 9.870316007963952e-19]
p(145) ∈ [0.0, 9.870316007963952e-19]
p(146) ∈ [0.0, 9.870316007963952e-19]
p(147) ∈ [0.0, 9.870316007963952e-19]
p(148) ∈ [0.0, 9.870316007963952e-19]
p(149) ∈ [0.0, 9.870316007963952e-19]
p(150) ∈ [0.0, 9.870316007963952e-19]
p(151) ∈ [0.0, 9.870316007963952e-19]
p(152) ∈ [0.0, 9.870316007963952e-19]
p(153) ∈ [0.0, 9.870316007963952e-19]
p(154) ∈ [0.0, 9.870316007963952e-19]
p(155) ∈ [0.0, 9.870316007963952e-19]
p(156) ∈ [0.0, 9.870316007963952e-19]
p(157) ∈ [0.0, 9.870316007963952e-19]
p(158) ∈ [0.0, 9.870316007963952e-19]
p(159) ∈ [0.0, 9.870316007963952e-19]
p(160) ∈ [0.0, 9.870316007963952e-19]
p(161) ∈ [0.0, 9.870316007963952e-19]
p(162) ∈ [0.0, 9.870316007963952e-19]
p(163) ∈ [0.0, 9.870316007963952e-19]
p(164) ∈ [0.0, 9.870316007963952e-19]
p(165) ∈ [0.0, 9.870316007963952e-19]
p(166) ∈ [0.0, 9.870316007963952e-19]
p(167) ∈ [0.0, 9.870316007963952e-19]
p(168) ∈ [0.0, 9.870316007963952e-19]
p(169) ∈ [0.0, 9.870316007963952e-19]
p(170) ∈ [0.0, 9.870316007963952e-19]
p(171) ∈ [0.0, 9.870316007963952e-19]
p(172) ∈ [0.0, 9.870316007963952e-19]
p(173) ∈ [0.0, 9.870316007963952e-19]
p(174) ∈ [0.0, 9.870316007963952e-19]
p(175) ∈ [0.0, 9.870316007963952e-19]
p(176) ∈ [0.0, 9.870316007963952e-19]
p(177) ∈ [0.0, 9.870316007963952e-19]
p(178) ∈ [0.0, 9.870316007963952e-19]
p(179) ∈ [0.0, 9.870316007963952e-19]
p(180) ∈ [0.0, 9.870316007963952e-19]
p(181) ∈ [0.0, 9.870316007963952e-19]
p(182) ∈ [0.0, 9.870316007963952e-19]
p(183) ∈ [0.0, 9.870316007963952e-19]
p(184) ∈ [0.0, 9.870316007963952e-19]
p(185) ∈ [0.0, 9.870316007963952e-19]
p(186) ∈ [0.0, 9.870316007963952e-19]
p(187) ∈ [0.0, 9.870316007963952e-19]
p(188) ∈ [0.0, 9.870316007963952e-19]
p(189) ∈ [0.0, 9.870316007963952e-19]
p(190) ∈ [0.0, 9.870316007963952e-19]
p(191) ∈ [0.0, 9.870316007963952e-19]
p(192) ∈ [0.0, 9.870316007963952e-19]
p(193) ∈ [0.0, 9.870316007963952e-19]
p(194) ∈ [0.0, 9.870316007963952e-19]
p(195) ∈ [0.0, 9.870316007963952e-19]
p(196) ∈ [0.0, 9.870316007963952e-19]
p(197) ∈ [0.0, 9.870316007963952e-19]
p(198) ∈ [0.0, 9.870316007963952e-19]
p(199) ∈ [0.0, 9.870316007963952e-19]
p(n) ∈ [0.0, 2.467579001990988e-19] for all n >= 200

Moments:
0-th (raw) moment ∈ [0.9999999999999999, 1.0000000000000002]
1-th (raw) moment ∈ [1.4999999999999998, inf]
2-th (raw) moment ∈ [2.9999999999999996, inf]
3-th (raw) moment ∈ [8.249999999999982, inf]
4-th (raw) moment ∈ [29.999999999999332, inf]
Total time: 0.00972s
"""                          

indices = list(range(200))[1:100]

# Extracted bounds for rest_output
rest_bounds = parse_output(rest_output)[1:100]
rest_lowers = [lower for lower, _ in rest_bounds]
rest_uppers = [upper for _, upper in rest_bounds]

# Extracted bounds for geom_bound_output
geom_bounds = parse_output(geom_bound_output)[1:100]
geom_bound_lowers = [lower for lower, _ in geom_bounds]
geom_bound_uppers = [upper for _, upper in geom_bounds]

# Extracted bounds for geom_bound tail output
tail_bounds = parse_output(tail_output)[1:100]
tail_uppers = [upper for _, upper in tail_bounds]

# Exact solution
exact = [2/3**i if i > 0 else 0 for i in indices]

fig, ax = plt.subplots(figsize=(6, 4))

# Plot rest bounds:
ax.plot(indices, rest_lowers, 'r--', marker='|', alpha=0.5, linewidth=1)
ax.plot(indices, rest_uppers, 'r-', marker='|', alpha=0.5, linewidth=1)
ax.fill_between(indices, rest_lowers, rest_uppers, color='red', alpha=0.2, label='Residual Mass Bounds')

# Plot geom_bound bounds:
ax.plot(indices, geom_bound_lowers, 'b-', marker='|', alpha=0.5, linewidth=1)
ax.plot(indices, geom_bound_uppers, 'b--', marker='|', alpha=0.5, linewidth=1)
ax.fill_between(indices, geom_bound_lowers, geom_bound_uppers, color='blue', alpha=0.2, label='Geometric Bound')

# Plot tail bounds:
ax.plot(indices, tail_uppers, 'k-', marker='|', alpha=0.5, linewidth=1, label='Geometric Bound (tail-optimized)')

# Plot exact masses:
ax.scatter(indices, exact, marker='x', color='green', zorder=5, s=20, label='Exact probability')

# Setting symmetrical logarithmic scale
linthresh = 1e-48
ax.set_yscale('symlog', linthresh=linthresh)

# Customizing the y-axis labels
ax.set_ylim(bottom=0, top=1)
ax.set_yticks([0, 1e-48, 1e-40, 1e-32, 1e-24, 1e-16, 1e-8, 1])
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

plt.xlabel('Result value')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# save as pdf
plt.savefig('rest_vs_geom_bound_die_paradox.pdf', bbox_inches='tight')

plt.show()