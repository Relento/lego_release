from functools import partial, reduce
from typing import List, Any, Union


def map_nested_list(fn, *list_of_nested_list: List[List[Union[Any, List[Any]]]],
                    template: List[Union[Any, List[Any]]]) -> List[Union[Any, List[Any]]]:
    """

    Args:
        fn:
        template: WARNINGS: 'Any' cannot be a list
        *list_of_nested_list:
            list_of_nested_list[0], list_of_nested_list[1], ... should have the same nested structure as `template`

    Returns:

    """
    for nested_list in list_of_nested_list:
        assert len(template) == len(nested_list), (template, nested_list)
    # first flatten all nested_list
    list_of_flattened_nested_list = map(partial(flatten_nested_list, template=template), list_of_nested_list)
    # then use the regular map function
    # try:
    flattened_ret = list(map(lambda arg_tuple: fn(*arg_tuple), zip(*list_of_flattened_nested_list)))
    # except:
    #     print('using template for map nested', template)
    #     print(list_of_nested_list)
    #     import ipdb; ipdb.set_trace()
    #     print()
    # then unflatten using the template
    return unflatten_nested_list(flattened_ret, template=template)

    ret = []

    for i, item in enumerate(zip(*list_of_nested_list)):
        if isinstance(template[i], list):
            assert len(item[0]) == len(template[i])
            # item[0] is a list
            ret.append([])
            for inner_item in zip(*item):
                ret[-1].append(fn(*inner_item))
        else:
            ret.append(fn(*item))

    return ret


def reduce_nested_list(reduce_fn, nested_list, template):
    # apply reduce_op to nested entries in `nested_list`
    # returned list will be a plain list of length `len(nested_list)`
    assert len(template) == len(nested_list), (len(nested_list), len(template))
    ret = []
    for i in range(len(template)):
        if isinstance(template[i], list):
            ret.append(reduce(reduce_fn, nested_list[i]))
        else:
            ret.append(nested_list[i])
    return ret


def flatten_nested_list(nested_list, template) -> List[Any]:
    assert len(template) == len(nested_list), (len(nested_list), len(template))
    ret = []
    for i in range(len(template)):
        if isinstance(template[i], list):
            ret.extend(nested_list[i])
        else:
            ret.append(nested_list[i])
    return ret


def unflatten_nested_list(flattend_nested_list, template) -> List[Union[Any, List[Any]]]:
    ret = []
    j = 0
    for i in range(len(template)):
        if isinstance(template[i], list):
            new_j = j + len(template[i])
            ret.append(flattend_nested_list[j:new_j])
            j = new_j
        else:
            ret.append(flattend_nested_list[j])
            j = j + 1

    return ret


def index_list(l, idxs, check=True):
    if check:
        if list(idxs) != sorted(list(idxs)):
            print('[WARNINGS] indices are not sorted, this might lead to unexpected behaviors')
            # import ipdb; ipdb.set_trace()

    # return [l[i] for i in sorted(list(idxs))]
    return [l[i] for i in list(idxs)]
