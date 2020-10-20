import pandas as pd


class ProbEvent:
    def __init__(self, variables: dict, conditions: dict):
        self.vars = dict()
        self.conds = dict()

        for k in sorted(variables.keys()):
            self.vars[k] = variables[k]
        if conditions is not None:
            for k in sorted(conditions.keys()):
                self.conds[k] = conditions[k]

    def show(self):
        return str(self.vars) + str(self.conds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(str(self.vars) + str(self.conds))


class ProbTable:
    def __init__(self, data):
        self.data = data
        self.table = dict()

    def get_probabilty(self, variables, conditions=None):
        assert isinstance(variables, dict)
        if conditions:
            assert isinstance(conditions, dict)

        event = ProbEvent(variables, conditions)
        if event not in self.table:
            self.comput_probability(event)
        return self.table[event]

    def comput_probability(self, event):
        cond_data = self.data
        if event.conds is not None:
            for key, val in event.conds.items():
                cond_data = cond_data[cond_data[key] == val]
        denominator = len(cond_data)
        assert denominator != 0

        for key, val in event.vars.items():
            cond_data = cond_data[cond_data[key] == val]
        numerator = len(cond_data)

        self.table[event] = numerator / denominator


def compute_probs(data):
    probs = pd.DataFrame()
    pt = ProbTable(data)

    for row in data.itertuples():
        row_dict = {}

        # --------- R1 -----------------
        p1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})
        p0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})


        row_dict['R1_1'] = p1
        row_dict['R1_0'] = p0

        # --------- T1 -----------------
        n1 = pt.get_probabilty(variables={'s': 1}, conditions={'x1': getattr(row, 'x1')})
        n0 = pt.get_probabilty(variables={'s': 0}, conditions={'x1': getattr(row, 'x1')})

        s1 = pt.get_probabilty(variables={'s': 1})
        s0 = pt.get_probabilty(variables={'s': 0})

        row_dict['T1_1'] = n1 / s1
        row_dict['T1_0'] = n0 / s0

        # -------- R2 --------------------
        p11 = pt.get_probabilty(variables={'y2': 1}, conditions={'y1': 1, 'x2': getattr(row, 'x2')})
        p01 = pt.get_probabilty(variables={'y2': 1}, conditions={'y1': 0, 'x2': getattr(row, 'x2')})
        p10 = pt.get_probabilty(variables={'y2': 0}, conditions={'y1': 1, 'x2': getattr(row, 'x2')})
        p00 = pt.get_probabilty(variables={'y2': 0}, conditions={'y1': 0, 'x2': getattr(row, 'x2')})

        n1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})

        d1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})
        d0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})


        row_dict['R2_11'] = p11 * n1 / d1
        row_dict['R2_10'] = p10 * n1 / d1
        row_dict['R2_01'] = p01 * n0 / d0
        row_dict['R2_00'] = p00 * n0 / d0

        # --------- T2 --------------------
        n11 = pt.get_probabilty(variables={'s': 1, 'y1': 1}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n10 = pt.get_probabilty(variables={'s': 1, 'y1': 0}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n01 = pt.get_probabilty(variables={'s': 0, 'y1': 1}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n00 = pt.get_probabilty(variables={'s': 0, 'y1': 0}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})

        d11 = pt.get_probabilty(variables={'s': 1, 'y1': 1}, conditions={'x1': getattr(row, 'x1')})
        d10 = pt.get_probabilty(variables={'s': 1, 'y1': 0}, conditions={'x1': getattr(row, 'x1')})
        d01 = pt.get_probabilty(variables={'s': 0, 'y1': 1}, conditions={'x1': getattr(row, 'x1')})
        d00 = pt.get_probabilty(variables={'s': 0, 'y1': 0}, conditions={'x1': getattr(row, 'x1')})

        row_dict['T2_11'] = row_dict['T1_1'] * n11 / d11
        row_dict['T2_10'] = row_dict['T1_1'] * n10 / d10
        row_dict['T2_01'] = row_dict['T1_0'] * n01 / d01
        row_dict['T2_00'] = row_dict['T1_0'] * n00 / d00
        
        probs = probs.append(row_dict, ignore_index=True)

    return probs


def compute_probs_adult(data):
    probs = pd.DataFrame()
    pt = ProbTable(data)

    for row in data.itertuples():
        row_dict = {}

        # --------- R1 -----------------
        p1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})
        p0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})

        row_dict['R1_1'] = p1
        row_dict['R1_0'] = p0
        
        # --------- T1 -----------------
        n1 = pt.get_probabilty(variables={'s': 1}, conditions={'x1': getattr(row, 'x1')})
        n0 = pt.get_probabilty(variables={'s': 0}, conditions={'x1': getattr(row, 'x1')})

        s1 = pt.get_probabilty(variables={'s': 1})
        s0 = pt.get_probabilty(variables={'s': 0})

        row_dict['T1_1'] = n1 / s1
        row_dict['T1_0'] = n0 / s0
        
        # # -------- R2 --------------------
        p11 = pt.get_probabilty(variables={'y2': 1}, conditions={'y1': 1, 's': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        p01 = pt.get_probabilty(variables={'y2': 1}, conditions={'y1': 0, 's': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        p10 = pt.get_probabilty(variables={'y2': 0}, conditions={'y1': 1, 's': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        p00 = pt.get_probabilty(variables={'y2': 0}, conditions={'y1': 0, 's': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})

        n1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})

        d1 = pt.get_probabilty(variables={'y1': 1}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})
        d0 = pt.get_probabilty(variables={'y1': 0}, conditions={'s': getattr(row, 's'), 'x1': getattr(row, 'x1')})

        row_dict['R2_11'] = p11 * n1 / d1
        row_dict['R2_10'] = p10 * n1 / d1
        row_dict['R2_01'] = p01 * n0 / d0
        row_dict['R2_00'] = p00 * n0 / d0

        # # --------- T2 --------------------
        n11 = pt.get_probabilty(variables={'s': 1, 'y1': 1}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n10 = pt.get_probabilty(variables={'s': 1, 'y1': 0}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n01 = pt.get_probabilty(variables={'s': 0, 'y1': 1}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})
        n00 = pt.get_probabilty(variables={'s': 0, 'y1': 0}, conditions={'x1': getattr(row, 'x1'), 'x2': getattr(row, 'x2')})

        d11 = pt.get_probabilty(variables={'s': 1, 'y1': 1}, conditions={'x1': getattr(row, 'x1')})
        d10 = pt.get_probabilty(variables={'s': 1, 'y1': 0}, conditions={'x1': getattr(row, 'x1')})
        d01 = pt.get_probabilty(variables={'s': 0, 'y1': 1}, conditions={'x1': getattr(row, 'x1')})
        d00 = pt.get_probabilty(variables={'s': 0, 'y1': 0}, conditions={'x1': getattr(row, 'x1')})

        row_dict['T2_11'] = row_dict['T1_1'] * n11 / d11
        row_dict['T2_10'] = row_dict['T1_1'] * n10 / d10
        row_dict['T2_01'] = row_dict['T1_0'] * n01 / d01
        row_dict['T2_00'] = row_dict['T1_0'] * n00 / d00

        probs = probs.append(row_dict, ignore_index=True)

    return probs