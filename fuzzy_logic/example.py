from pprint import pprint

from numpy import array as np_array
from skfuzzy import zmf, smf, trapmf

from fuzzy_logic.multi_criteria_affiliation import \
    MultiCriteriaAffiliationSelector, Criteria


class Cost(Criteria):
    CHEAP = 'Cheap'
    NORMAL = 'Normal'
    EXPENSIVE = 'Expensive'

    @staticmethod
    def get_fuzzy_membership_map():
        normal_trapezoid = np_array([120, 280, 320, 500])

        return {
            Cost.CHEAP: lambda x: zmf(x, a=0, b=180),
            Cost.NORMAL: lambda x: trapmf(x, abcd=normal_trapezoid),
            Cost.EXPENSIVE: lambda x: smf(x, a=300, b=1000)
        }


class Distance(Criteria):
    SHORT = 'Short'
    AVERAGE = 'Average'
    LONG = 'Long'

    @staticmethod
    def get_fuzzy_membership_map():
        average_trapezoid = np_array([10, 48, 52, 90])

        return {
            Distance.SHORT: lambda x: zmf(x, a=0, b=30),
            Distance.AVERAGE: lambda x: trapmf(x, abcd=average_trapezoid),
            Distance.LONG: lambda x: smf(x, a=20, b=100)
        }


class Free(Criteria):
    FREE = 'Free'
    NOT_FREE = 'Not Free'

    @staticmethod
    def get_fuzzy_membership_map():
        return {
            Free.FREE: lambda x: zmf(x, a=0, b=6),
            Free.NOT_FREE: lambda x: smf(x, a=2, b=10)
        }


class Comfort(Criteria):
    COMFORT = 'Comfort'
    NOT_COMFORT = 'Not Comfort'

    @staticmethod
    def get_fuzzy_membership_map():
        comfort_trapezoid = np_array([0, 0.3, 0.4, 0.6])

        return {
            Comfort.COMFORT: lambda x: trapmf(x, abcd=comfort_trapezoid),
            Comfort.NOT_COMFORT: lambda x: smf(x, a=0.4, b=1)
        }


EXPERT_MATRIX = {
    (Cost.CHEAP, Distance.SHORT, Free.FREE): Comfort.COMFORT,
    (Cost.CHEAP, Distance.SHORT, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.CHEAP, Distance.AVERAGE, Free.FREE): Comfort.COMFORT,
    (Cost.CHEAP, Distance.AVERAGE, Free.NOT_FREE): Comfort.COMFORT,
    (Cost.CHEAP, Distance.LONG, Free.FREE): Comfort.COMFORT,
    (Cost.CHEAP, Distance.LONG, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.NORMAL, Distance.SHORT, Free.FREE): Comfort.COMFORT,
    (Cost.NORMAL, Distance.SHORT, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.NORMAL, Distance.AVERAGE, Free.FREE): Comfort.COMFORT,
    (Cost.NORMAL, Distance.AVERAGE, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.NORMAL, Distance.LONG, Free.FREE): Comfort.COMFORT,
    (Cost.NORMAL, Distance.LONG, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.EXPENSIVE, Distance.SHORT, Free.FREE): Comfort.NOT_COMFORT,
    (Cost.EXPENSIVE, Distance.SHORT, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.EXPENSIVE, Distance.AVERAGE, Free.FREE): Comfort.NOT_COMFORT,
    (Cost.EXPENSIVE, Distance.AVERAGE, Free.NOT_FREE): Comfort.NOT_COMFORT,
    (Cost.EXPENSIVE, Distance.LONG, Free.FREE): Comfort.COMFORT,
    (Cost.EXPENSIVE, Distance.LONG, Free.NOT_FREE): Comfort.NOT_COMFORT,
}


def demonstrate():
    num_of_samples = 21
    criteria_values = {
        Cost: 150,
        Distance: 30,
        Free: 4
    }

    affiliation_selector = MultiCriteriaAffiliationSelector(
        criteria_values, EXPERT_MATRIX, num_of_samples)

    terms = affiliation_selector.get_terms()
    involved_rules = affiliation_selector.select_involved_rules(terms)
    membership_criteria = affiliation_selector.membership_criteria()
    result_criterion = affiliation_selector.affiliation_criterion()

    print('Involved terms:')
    pprint(terms)
    print('Selected rules:')
    pprint(involved_rules)
    print('Values of the affirmation criteria:')
    pprint(membership_criteria)

    user_result_message = 'Travel is: {}'.format(result_criterion.value)
    print(user_result_message)
