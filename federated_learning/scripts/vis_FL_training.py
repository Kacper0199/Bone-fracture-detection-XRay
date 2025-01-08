import matplotlib.pyplot as plt

def calculate_averages(list_of_lists):
    averages = []
    for lst in list_of_lists:
        if len(lst) > 0:
            avg = sum(lst) / len(lst)
            averages.append(avg)
    return averages

gauss_e5 = [
    [2.3323766867319744, 2.2673484760781992],
    [2.119014096260071, 2.0360129527423694],
    [2.067687541908688, 1.9116838937220366],
    [1.9516634795400831, 1.765568891297216],
    [1.7969520078765022, 1.5960698853368345],
    [1.649657052093082, 1.4258074941842451],
    [1.4754077355066935, 1.2198196597721265],
    [1.2656178149912092, 0.9682068539702374],
    [1.124092957046297, 0.8863164341968038],
    [1.1690249396695032, 1.0911657214164734]
        ]

gauss_e4 = [
    [2.4638882637023927, 2.4333974174831225],
    [2.2633631812201607, 2.2165952039801557],
    [2.084400310781267, 2.0125689895256706],
    [1.9734557708104452, 1.8620654163153276],
    [1.8596515244907803, 1.7220587756322778],
    [1.7483054134580824, 1.5541522269663604],
    [1.6123671611150105, 1.374737949474998],
    [1.4357198887401157, 1.1454017447388691],
    [1.2005422512690227, 0.8482281863689423],
    [1.2062494913736979, 1.0470114155955936] 
]
gauss_e3 = [
    [2.7040843539767794, 2.698485851287842],
    [2.5237837658988105, 2.5073374976282534],
    [2.3432415458891125, 2.306711440501006],
    [2.1886245965957642, 2.142569632633873],
    [2.073545510239071, 2.0025346616040105],
    [1.9724258250660367, 1.8739305086757825],
    [1.8950763715638055, 1.7543232492778613],
    [1.799830444653829, 1.6079972438190295],
    [1.6579489999347263, 1.4033952433130015],
    [1.4748017403814528, 1.1811227824376977]
]
    
rounds = range(1, len(gauss_e3) + 1)
plt.figure(figsize=(10, 5))
plt.plot(rounds, calculate_averages(gauss_e5), label=r'|$\epsilon$| = 1e-5', marker='o')
plt.plot(rounds, calculate_averages(gauss_e4), label=r'|$\epsilon$| = 1e-4', marker='o')
plt.plot(rounds, calculate_averages(gauss_e3), label=r'|$\epsilon$| = 1e-3', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Evaluation Metric')
plt.title(r'Validation loss over epochs for different $\epsilon$ values')
plt.legend()
plt.grid(True)
plt.show()

def plot_results(to_vis, eps):

    do1_val_eval = [x[0] for x in to_vis]
    bfy_val_eval = [x[1] for x in to_vis]
    rounds = range(1, len(to_vis) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, do1_val_eval, label='do1 validation eval', marker='o')
    plt.plot(rounds, bfy_val_eval, label='bone-fracture-yzkpo validation eval', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Metric')
    plt.title(fr'Validation loss over epochs for different datasets with $\epsilon$ = {eps}')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_results(gauss_e3, "1e-3")
plot_results(gauss_e4, "1e-4")
plot_results(gauss_e5, "1e-5")