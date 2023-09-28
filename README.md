# DRL urban planning
![Loading Model Overview](assets/pipeline_v3.png "Model Overview")
---

In this project, we propose a reinforcement-learning-based framework for assisting urban planners in the complex task of optimizing the spatial design of urban communities.
Our proposed model can generate land and road layout with superior spatial efficiency, and improve the productivity of human planners with a human-AI collaborative workflow.

This project was initially described in the [research article in *Nature Computational Science*](https://www.nature.com/articles/s43588-023-00503-5):

Yu Zheng, Yuming Lin, Liang Zhao, Tinghai Wu, Depeng Jin, Yong Li. **Spatial planning of urban communities via deep reinforcement learning**. Nat Comput Sci (2023). https://doi.org/10.1038/s43588-023-00503-5

Full text (PDF) is available at [this link](https://rdcu.be/dlRPZ).

## Installation 

### Environment
* **Tested OS:** Linux
* Python >= 3.8
* PyTorch >= 1.8.1, <= 1.13.0
### Dependencies:
1. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) with the correct CUDA version.
2. Set the following environment variable to avoid problems with multiprocess trajectory sampling:
    ```
    export OMP_NUM_THREADS=1
    ```
   
## Data
The data used for training and evaluation can be found in [urban_planning/cfg/test_data](urban_planning/cfg/test_data).
We provide all the three scenarios used in our paper, including one synthetic grid community in [urban_planning/cfg/test_data/synthetic](urban_planning/cfg/test_data/synthetic), and two real-world communities, HLG and DHM, with and without planning concepts, in [urban_planning/cfg/test_data/real](urban_planning/cfg/test_data/real).
The data for the real-world communities are collected from the widely used [OpenStreetMap](https://www.openstreetmap.org/) (OSM) using [OSMnx](https://github.com/gboeing/osmnx).
For each case, we provide the following data:
* `init_plan.pickle`: the initial conditions of the community in [geopandas.GeoDataFrame](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html) form, including the initial land blocks, roads, and junctions.
* `objectives.yaml`: the planning objectives (requirements) of the community in [yaml](https://yaml.org/) form, including the required number/area of different functionalities, and the minimum/maximum area of each land use type.

The figure below illustrates the initial conditions of the three scenarios.
![Loading Data Overview](assets/initial_conditions.png "Initial Conditions")

With the initial conditions and planning objectives, the agent will generate millions of spatial plans for the community in real-time during training, which are stored in the replay buffer for training.

## Training
You can train your own models using the provided config in [urban_planning/cfg/exp_cfg/real](urban_planning/cfg/exp_cfg/real).

For example, to train a model for the HLG community, run:
```
python3 -m urban_planning.train --cfg hlg --global_seed 111
```
You can replace `hlg` to `dhm` to train for the DHM community.

To train a model with planning concepts for the HLG community, run:
``` 
python3 -m urban_planning.train --cfg hlg_concept --global_seed 111
```
You can replace `hlg_concept` to `dhm_concept` to train for the DHM community.

You will see the following logs once you start training our model:

https://user-images.githubusercontent.com/27959377/242561690-c01480a2-cedf-4889-8506-14add002227a.mp4

## Visualization
You can visualize the generated spatial plans using the provided notebook in [demo](demo).

## Baselines
To evaluate the centralized heuristic, run:
```
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent rule-centralized
```

To evaluate the decentralized heuristic, run:
```
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent rule-decentralized
```

To evaluate the geometric set-coverage adapted baseline, run:
```
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent gsca
```

To evaluate the GA baseline, run:
```
python3 -m urban_planning.train_ga --cfg hlg --global_seed 111
python3 -m urban_planning.eval --cfg hlg --global_seed 111 --agent ga
```
You can replace `hlg` to `dhm` to evaluate for the DHM community.

## License
Please see the [license](LICENSE) for further details.

## Awesome AI Planning Support Tools
With urban planning being a long-standing problem, researchers have devoted decades of efforts to developing computational models and supporting tools for it in order to automate its process. 
Automated spatial layout seemed impossible until much more recently with the latest advancements in artificial intelligence, especially deep reinforcement learning.
In fact, our proposed DRL approach is inspired by, and takes a small step forward from, the planning support tools that have been available for the past few decades.
Here, we summarize these existing awesome planning support tools that utilize AI to facilitate urban planning.

### Land use planning

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Wang, D., Fu, Y., Wang, P., Huang, B., & Lu, C. T. (2020, November). Reimagining city configuration: Automated urban planning via adversarial learning. In Proceedings of the 28th international conference on advances in geographic information systems(pp. 497-506).](https://dl.acm.org/doi/abs/10.1145/3397536.3422268)                                                                                                                                                                                                                                                     | land-use configuration, GAN                    | SIGSPATIAL                                                   |   2020 |
| [Wang, D., Fu, Y., Liu, K., Chen, F., Wang, P., & Lu, C. T. (2023). Automated urban planning for reimagining city configuration via adversarial learning: quantification, generation, and evaluation.ACM Transactions on Spatial Algorithms and Systems,9(1), 1-24.](https://dl.acm.org/doi/10.1145/3524302)                                                                                                                                                                                                                                                                     | land-use configuration, GAN                    | ACM Transactions on Spatial Algorithms and Systems           |   2023 |
| [Wang, D., Wu, L., Zhang, D., Zhou, J., Sun, L., & Fu, Y. (2023, June). Human-instructed Deep Hierarchical Generative Learning for Automated Urban Planning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 4, pp. 4660-4667).](https://ojs.aaai.org/index.php/AAAI/article/view/25589)                                                                                                                                                                                                                                                          | land-use configuration, Transformer            | AAAI                                                          |   2023 |
| [Yao, J., Murray, A. T., Wang, J., & Zhang, X. (2019). Evaluation and development of sustainable urban land use plans through spatial optimization.Transactions in GIS,23(4), 705-725.](https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.12531)                                                                                                                                                                                                                                                                                                                             | land use plan                                  | Transactions in GIS                                          |   2019 |
| [Dahal, K. R., & Chow, T. E. (2014). A GIS toolset for automated partitioning of urban lands.Environmental Modelling & Software,55, 222-234.](https://www.sciencedirect.com/science/article/pii/S1364815214000346)                                                                                                                                                                                                                                                                                                                                                               | land use partition, ArcGIS toolset             | Environmental Modelling & Software                           |   2014 |
| [Ligmann‐Zielinska, A., Church, R. L., & Jankowski, P. (2008). Spatial optimization as a generative technique for sustainable multiobjective land‐use allocation. International Journal of Geographical Information Science, 22(6), 601-622.](https://www.tandfonline.com/doi/abs/10.1080/13658810701587495)                                                                                                                                                                                                                                                                     | land-use allocation, optimization              | International Journal of Geographical Information Science    |   2008 |
| [Xu, Y., Olmos, L. E., Abbar, S., & González, M. C. (2020). Deconstructing laws of accessibility and facility distribution in cities.Science advances,6(37), eabb4112.](https://www.science.org/doi/full/10.1126/sciadv.abb4112)                                                                                                                                                                                                                                                                                                                                                 | facility location, swap-based local search     | Science Advances                                             |   2020 |
| [Merrell, P., Schkufza, E., & Koltun, V. (2010). Computer-generated residential building layouts. InACM SIGGRAPH Asia 2010 papers(pp. 1-12).](https://dl.acm.org/doi/abs/10.1145/1866158.1866203)                                                                                                                                                                                                                                                                                                                                                                                | building layout,Bayesian network               | ACM SIGGRAPH Asia                                            |   2010 |
| [Liu, Y., Luo, Y., Deng, Q., & Zhou, X. (2021). Exploration of Campus Layout Based on Generative Adversarial Network: Discussing the Significance of Small Amount Sample Learning for Architecture. In Proceedings of the 2020 DigitalFUTURES: The 2nd International Conference on Computational Design and Robotic Fabrication (CDRF 2020) (pp. 169-178). Springer Singapore.](https://link.springer.com/chapter/10.1007/978-981-33-4400-6_16)                                                                                                                                  | campus layout, GAN                             | Digital Futures                                              |   2021 |
| [Tian, R. (2021). Suggestive site planning with conditional gan and urban gis data. In Proceedings of the 2020 DigitalFUTURES: The 2nd International Conference on Computational Design and Robotic Fabrication (CDRF 2020) (pp. 103-113). Springer Singapore.](https://link.springer.com/chapter/10.1007/978-981-33-4400-6_10)                                                                                                                                                                                                                                                  | site planning, GAN                             | Digital Futures                                              |   2020 |

### Road planning

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Zheng, Y., Su, H., Ding, J., Jin, D., & Li, Y. (2023). Road Planning for Slums via Deep Reinforcement Learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 5695–5706).](https://dl.acm.org/doi/10.1145/3580305.3599901)                                                                                                                                                                                                                                                                                                                 | road network generation, DRL                 | KDD                                  |   2023 |
| [Xue, J., Jiang, N., Liang, S., Pang, Q., Yabe, T., Ukkusuri, S. V., & Ma, J. (2022). Quantifying the spatial homogeneity of urban road networks via graph neural networks. Nature Machine Intelligence, 4(3), 246-257.](https://www.nature.com/articles/s42256-022-00462-y)                                                                                                                                                                                                                                                                                                               | road network analysis, GNN                 | Nature Machine Intelligence                                  |   2022 |
| [Fang, Z., Jin, Y., & Yang, T. (2022). Incorporating planning intelligence into deep learning: A planning support tool for street network design.Journal of Urban Technology,29(2), 99-114.](https://www.tandfonline.com/doi/full/10.1080/10630732.2021.2001713)                                                                                                                                                                                                                                                                                                                 | street network generation, GAN                 | Journal of Urban Technology                                  |   2022 |
| [Fang, Z., Qi, J., Fan, L., Huang, J., Jin, Y., & Yang, T. (2022). A topography-aware approach to the automatic generation of urban road networks.International Journal of Geographical Information Science,36(10), 2035-2059.](https://doi.org/10.1080/13658816.2022.2072849)                                                                                                                                                                                                                                                                                                   | road network generation, GAN                   | International Journal of Geographical Information Science    |   2022 |
| [Li, J., Lyu, L., Shi, J., Zhao, J., Xu, J., Gao, J., ... & Sun, Z. (2022, November). Generating community road network from GPS trajectories via style transfer. In Proceedings of the 30th International Conference on Advances in Geographic Information Systems (pp. 1-4).](https://dl.acm.org/doi/10.1145/3557915.3560958)                                                                                                                                                                                                                                                  | road network generation, Style Transfer        | SIGSPATIAL                                                   |   2022 |

### Urban design

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Zheng, H., & Yuan, P. F. (2021). A generative architectural and urban design method through artificial neural networks. Building and Environment, 205, 108178.](https://www.sciencedirect.com/science/article/pii/S0360132321005795)                                                                                                                                                                                                                                                                                                                                            | urban design, ANN                              | Building and Environment                                     |   2021 |
| [Sun, Y., & Dogan, T. (2023). Generative methods for Urban design and rapid solution space exploration.Environment and Planning B: Urban Analytics and City Science,50(6), 1577-1590.](https://journals.sagepub.com/doi/abs/10.1177/23998083221142191)                                                                                                                                                                                                                                                                                                                           | urban design, tensor field                     | Environment and Planning B: Urban Analytics and City Science |   2023 |
| [Duering, S., Chronic, A., & Koenig, R. (2020, May). Optimizing Urban Systems: Integrated optimization of spatial configurations. In Proceedings of the 11th annual symposium on simulation for architecture and urban design (pp. 1-7).](http://simaud.org/2020/proceedings/109.pdf)                                                                                                                                                                                                                                                                                            | urban design, evolutionary optimization        | SimAUD                                                       |   2020 |
| [Han, Z., Yan, W., & Liu, G. (2021). A performance-based urban block generative design using deep reinforcement learning and computer vision. InProceedings of the 2020 DigitalFUTURES: The 2nd International Conference on Computational Design and Robotic Fabrication (CDRF 2020)(pp. 134-143). Springer Singapore.](https://link.springer.com/chapter/10.1007/978-981-33-4400-6_13)                                                                                                                                                                                          | urban block design, deep RL, CV                | Digital Futures                                              |   2020 |
| [Fedorova, S. (2021). Generative adversarial networks for urban block design. InSimAUD 2021: A Symposium on Simulation for Architecture and Urban Design.](http://simaud.org/proceedings/repo/2021-Papers/38-Fedorova.pdf)                                                                                                                                                                                                                                                                                                                                                       | urban block design, GAN                        | SimAUD                                                       |   2021 |
| [Koenig, R., Miao, Y., Aichinger, A., Knecht, K., & Konieva, K. (2020). Integrating urban analysis, generative design, and evolutionary optimization for solving urban design problems. Environment and Planning B: Urban Analytics and City Science, 47(6), 997-1013.](https://journals.sagepub.com/doi/abs/10.1177/2399808319894986)                                                                                                                                                                                                                                           | urban master-design, evolutionary optimization | Environment and Planning B: Urban Analytics and City Science |   2019 |
| [Ye, X., Du, J., & Ye, Y. (2022). MasterplanGAN: Facilitating the smart rendering of urban master plans via generative adversarial networks.Environment and Planning B: Urban Analytics and City Science,49(3), 794-814.](https://journals.sagepub.com/doi/abs/10.1177/23998083211023516)                                                                                                                                                                                                                                                                                        | urban master-design, GAN                       | Environment and Planning B: Urban Analytics and City Science |   2021 |
| [Hua, H., Hovestadt, L., Tang, P., & Li, B. (2019). Integer programming for urban design.European Journal of Operational Research,274(3), 1125-1137.](https://www.sciencedirect.com/science/article/abs/pii/S0377221718309238)                                                                                                                                                                                                                                                                                                                                                   | urban design, integer programming              | European Journal of Operational Research                     |   2019 |
| [Hidalgo, C. A., Castañer, E., & Sevtsuk, A. (2020). The amenity mix of urban neighborhoods.Habitat International,106, 102205.](https://www.sciencedirect.com/science/article/abs/pii/S0197397519311336)                                                                                                                                                                                                                                                                                                                                                                         | urban design, amenities mix, clusters          | Habitat International                                        |   2020 |
| [Shen, J., Liu, C., Ren, Y., & Zheng, H. (2020, August). Machine learning assisted urban filling. InProceedings of the 25th CAADRIA Conference, Bangkok, Thailand(pp. 5-6).](https://papers.cumincad.org/data/works/att/caadria2020_054.pdf)                                                                                                                                                                                                                                                                                                                                     | urban filling, GAN                             | CAADRIA                                                      |   2020 |
| [Galanos, T., Liapis, A., Yannakakis, G. N., & Koenig, R. (2021, July). ARCH-Elites: Quality-diversity for urban design. InProceedings of the Genetic and Evolutionary Computation Conference Companion(pp. 313-314).](https://dl.acm.org/doi/abs/10.1145/3449726.3459490)                                                                                                                                                                                                                                                                                                       | urban design, MAP                              | GECCO                                                        |   2021 |
| [Xu, X., Yin, C., Wang, W., Xu, N., Hong, T., & Li, Q. (2019). Revealing urban morphology and outdoor comfort through genetic algorithm-driven urban block design in dry and hot regions of China.Sustainability,11(13), 3683.](https://www.mdpi.com/2071-1050/11/13/3683)                                                                                                                                                                                                                                                                                                       | urban block design, genetic algorithm          | Sustainability                                               |   2019 |

### 15-minute city

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Vich, G., Gómez-Varo, I., & Marquet, O. (2023). Measuring the 15-Minute City in Barcelona. A geospatial three-method comparison. In Resilient and Sustainable Cities(pp. 39-60). Elsevier.](https://www.sciencedirect.com/science/article/pii/B9780323917186000049)                                                                                                                                                                                                                                                                                                             | 15-minute city                                 | Resilient and Sustainable Cities                             |   2023 |
| [Ferrer-Ortiz, C., Marquet, O., Mojica, L., & Vich, G. (2022). Barcelona under the 15-minute city lens: mapping the accessibility and proximity potential based on pedestrian travel times.Smart Cities,5(1), 146-161.](https://www.mdpi.com/2624-6511/5/1/10)                                                                                                                                                                                                                                                                                                                   | 15-minute city, network analysis               | Smart Cities                                                 |   2022 |
| [Allam, Z., Nieuwenhuijsen, M., Chabaud, D., & Moreno, C. (2022). The 15-minute city offers a new framework for sustainability, liveability, and health.The Lancet Planetary Health,6(3), e181-e183.](https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(22)00014-6/fulltext)                                                                                                                                                                                                                                                                                       | 15-minute city                                 | The Lancet Planetary Health                                  |   2022 |
| [Allam, Z., Bibri, S. E., Chabaud, D., & Moreno, C. (2022). The ‘15-Minute City’concept can shape a net-zero urban future. Humanities and Social Sciences Communications, 9(1), 1-5.](https://www.nature.com/articles/s41599-022-01145-0)                                                                                                                                                                                                                                                                                                                                        | 15-minute city                                 | Humanities and Social Sciences Communications         |   2022 |
| [Noworól, A., Kopyciński, P., Hałat, P., Salamon, J., & Hołuj, A. (2022). The 15-Minute City—The Geographical Proximity of Services in Krakow. Sustainability, 14(12), 7103.](https://www.mdpi.com/2071-1050/14/12/7103)                                                                                                                                                                                                                                                                                                                                                         | 15-minute city, network analysis               | Sustainability                                               |   2022 |
| [Balletto, G., Ladu, M., Milesi, A., & Borruso, G. (2021). A methodological approach on disused public properties in the 15-minute city perspective.Sustainability,13(2), 593.](https://www.mdpi.com/2071-1050/13/2/593)                                                                                                                                                                                                                                                                                                                                                         | 15-minute city                                 | Sustainability                                               |   2021 |
| [Moreno, C., Allam, Z., Chabaud, D., Gall, C., & Pratlong, F. (2021). Introducing the “15-Minute City”: Sustainability, resilience and place identity in future post-pandemic cities. Smart Cities, 4(1), 93-111.](https://www.mdpi.com/2624-6511/4/1/6)                                                                                                                                                                                                                                                                                                                         | 15-minute city                                 | Smart Cities                                                 |   2021 |
| [Weng, M., Ding, N., Li, J., Jin, X., Xiao, H., He, Z., & Su, S. (2019). The 15-minute walkable neighborhoods: Measurement, social inequalities and implications for building healthy communities in urban China. Journal of Transport & Health, 13, 259-273.](https://www.sciencedirect.com/science/article/pii/S2214140518305103)                                                                                                                                                                                                                                              | 15-minute city                                 | Journal of Transport & Health                                |   2019 |

### Review

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Lin, B., Jabi, W., Corcoran, P., & Lannon, S. (2023). The application of deep generative models in urban form generation based on topology: a review. Architectural Science Review, 1-16.](https://www.tandfonline.com/doi/abs/10.1080/00038628.2023.2209550)                                                                                                                                                                                                                                                                                                                   | review, generative models, urban topology      | Architectural Science Review                                 |   2023 |
| [Miao, Y., Koenig, R., & Knecht, K. (2020, May). The development of optimization methods in generative urban design: a review. InProceedings of the 11th Annual Symposium on Simulation for Architecture and Urban Design(pp. 1-8).](http://simaud.org/2020/preprints/97.pdf)                                                                                                                                                                                                                                                                                                    | review, generative urban design, optimization  | SimAUD                                                       |   2020 |
| [Wu, A. N., Stouffs, R., & Biljecki, F. (2022). Generative Adversarial Networks in the built environment: A comprehensive review of the application of GANs across data types and scales. Building and Environment, 109477.](https://www.sciencedirect.com/science/article/abs/pii/S0360132322007089)                                                                                                                                                                                                                                                                            | review, GAN, built environment                 | Building and Environment                                     |   2023 |
| [Matthews, R. B., Gilbert, N. G., Roach, A., Polhill, J. G., & Gotts, N. M. (2007). Agent-based land-use models: a review of applications.Landscape Ecology,22, 1447-1459.](https://link.springer.com/article/10.1007/s10980-007-9135-1)                                                                                                                                                                                                                                                                                                                                         | review, land-use modeling                      | Landscape Ecology                                            |   2007 |

### Others

| **paper** | **keywords** | **venue** | **year** |
|---|---|---|---|
| [Quan, S. J. (2022). Urban-GAN: An artificial intelligence-aided computation system for plural urban design. Environment and Planning B: Urban Analytics and City Science, 49(9), 2500-2515.](https://journals.sagepub.com/doi/abs/10.1177/23998083221100550)                                                                                                                                                                                                                                                                                                                    | public participation, GAN                      | Environment and Planning B: Urban Analytics and City Science |   2022 |
| [Wang, X., Song, Y., & Tang, P. (2020). Generative urban design using shape grammar and block morphological analysis. Frontiers of Architectural Research, 9(4), 914-924.](https://www.sciencedirect.com/science/article/pii/S2095263520300662)                                                                                                                                                                                                                                                                                                                                  | shape grammar, CityEngine                      | Frontiers of Architectural Research                          |   2020 |
| [Rahimian, M., Beirão, J. N., Duarte, J. M. P., & Iulo, L. D. (2019). A Grammar-Based Generative Urban Design Tool Considering Topographic Constraints The Case for American Urban Planning. In37th Conference on Education and Research in Computer Aided Architectural Design in Europe and 23rd Conference of the Iberoamerican Society Digital Graphics, eCAADe SIGraDi 2019(pp. 267-276). Education and research in Computer Aided Architectural Design in Europe.](https://pure.psu.edu/en/publications/a-grammar-based-generative-urban-design-tool-considering-topograp) | shape grammar,topological constraint           | eCAADe SIGraDi                                                       |   2019 |
| [Huang, C., Zhang, G., Yao, J., Wang, X., Calautit, J. K., Zhao, C., ... & Peng, X. (2022). Accelerated environmental performance-driven urban design with generative adversarial network.Building and Environment,224, 109575.](https://www.sciencedirect.com/science/article/abs/pii/S0360132322008058)                                                                                                                                                                                                                                                                        | environmental performance, GAN                 | Building and Environment                                     |   2022 |
| [Sun, C., Zhou, Y., & Han, Y. (2022). Automatic generation of architecture facade for historical urban renovation using generative adversarial network.Building and Environment,212, 108781.](https://www.sciencedirect.com/science/article/pii/S0360132322000300)                                                                                                                                                                                                                                                                                                               | historical urban renovation, GAN               | Building and Environment                                     |   2022 |
| [Ligmann-Zielinska, A., & Jankowski, P. (2007). Agent-based models as laboratories for spatially explicit planning policies. Environment and Planning B: Planning and Design, 34(2), 316-335.](https://journals.sagepub.com/doi/10.1068/b32088)                                                                                                                                                                                                                                                                                                                                  | urban growth, ABM                              | Environment and Planning B: Urban Analytics and City Science |   2007 |
| [Zhang, W., Ma, Y., Zhu, D., Dong, L., & Liu, Y. (2022, August). Metrogan: Simulating urban morphology with generative adversarial network. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2482-2492).](https://dl.acm.org/doi/abs/10.1145/3534678.3539239)                                                                                                                                                                                                                                                                        | urban morphology, GAN                          | KDD                                                          |   2022 |
| [Jia, P., Fu, S., Li, Z., & He, H. (2019, December). Low-carbon optimization of spatial pattern in Shenfu New District based on genetic algorithm. In Journal of Physics: Conference Series (Vol. 1419, No. 1, p. 012039). IOP Publishing.](https://iopscience.iop.org/article/10.1088/1742-6596/1419/1/012039/meta)                                                                                                                                                                                                                                                             | spatial pattern, genetic algorithm, low-carbon | Journal of Physics: Conference Series                        |   2019 |
| [Balling, R. J., Taber, J. T., Brown, M. R., & Day, K. (1999). Multiobjective urban planning using genetic algorithm.Journal of urban planning and development,125(2), 86-99.](https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9488(1999)125:2(86))                                                                                                                                                                                                                                                                                                                           | urban planning, genetic algorithm              | Journal of Urban Planning and Development                    |   1999 |
