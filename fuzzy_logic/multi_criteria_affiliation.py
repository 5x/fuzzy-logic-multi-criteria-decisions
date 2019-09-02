from abc import ABCMeta, abstractmethod
from enum import Enum
from itertools import chain

import numpy as np

from fuzzy_logic.utils import rotate_matrix, contains_any_no_voids, \
    first_values_class


class Criteria(Enum):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def get_fuzzy_membership_map():
        raise NotImplementedError()

    @classmethod
    def get_involved_terms(cls, *values):
        actions_map = cls.get_fuzzy_membership_map()
        xs = np.array(values)
        involved_terms = set()

        for term, fuzzy_fn in actions_map.items():
            fuzzy_membership_values = fuzzy_fn(xs)

            if contains_any_no_voids(fuzzy_membership_values):
                involved_terms.add(term)

        return involved_terms

    def call_fuzzy_membership_fn(self, *values):
        xs = np.array(values)
        fuzzy_membership_fn = self.get_fuzzy_membership_map()[self]

        return fuzzy_membership_fn(xs)

    def get_sample_cut_of_ys(self, min_value, num_of_samples):
        xs = np.linspace(0, 1, num_of_samples)
        ys = self.call_fuzzy_membership_fn(*xs)

        return [y if y <= min_value else min_value for y in ys]


class MultiCriteriaAffiliationSelector(object):
    def __init__(self, criterion_values, expert_rules, num_of_samples):
        self.__criteria = criterion_values
        self.__expert_rules = expert_rules
        self.__num_of_samples = num_of_samples
        self.__affirmation_criteria = first_values_class(expert_rules)

    def membership_criteria(self):
        selected_terms = self.get_terms()
        involved_rules = self.select_involved_rules(selected_terms)
        affirmation_map = self.__build_affirmation_map(involved_rules)
        values = self.__calculate_affirmation_rule_values(affirmation_map)
        total_affirmation_value = self.__total_affirmation_value(values)

        return self.__get_membership_criteria(total_affirmation_value)

    def affiliation_criterion(self):
        criteria = self.membership_criteria()

        return max(criteria, key=criteria.get)

    def get_terms(self):
        terms = set()

        for criterion, value in self.__criteria.items():
            criterion_terms = criterion.get_involved_terms(value)
            terms.update(criterion_terms)

        return terms

    def select_involved_rules(self, involved_terms):
        return {rule_terms: affirmation
                for rule_terms, affirmation in self.__expert_rules.items()
                if len(set(rule_terms) - involved_terms) == 0}

    def __get_membership_criteria(self, value):
        affirmation_criteria = self.__affirmation_criteria
        criteria = {}

        for term in affirmation_criteria:
            values = affirmation_criteria.call_fuzzy_membership_fn(term, value)
            criteria[term], *_ = values

        return criteria

    def __get_min_membership_term_value(self, terms):
        values = (term.call_fuzzy_membership_fn(self.__criteria[type(term)])
                  for term in terms)

        return min(chain.from_iterable(values))

    def __build_affirmation_map(self, rules):
        affirmation_map = {}

        for rule_terms, affirmation_term in rules.items():
            min_value = self.__get_min_membership_term_value(rule_terms)
            value = affirmation_term.get_sample_cut_of_ys(
                min_value, self.__num_of_samples)

            affirmation_map[rule_terms] = value

        return affirmation_map

    @staticmethod
    def __calculate_affirmation_rule_values(affirmation_map):
        matrix = list(affirmation_map.values())

        return [max(i) for i in rotate_matrix(matrix)]

    @staticmethod
    def __total_affirmation_value(values):
        total = 0
        max_key_value = len(values) - 1

        for key, value in enumerate(values):
            total += ((key / max_key_value) * value)

        return total / sum(values)
