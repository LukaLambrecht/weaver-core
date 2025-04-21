# Studies of different resampling strategies

Resampling is handled mainly [here](https://github.com/LukaLambrecht/weaver-core/blob/bd998f4a52bef2672c84d794d0e9422eb67cdc04/weaver/utils/dataset.py#L57): instances are selected from the input data with a probability proportional to their assigned total weight.

The total weight can be the product of different factors, e.g.:
- cross-section and lumi weight
- class balancing weights
- phase space balancing weights

The latter are the subject of study here.
