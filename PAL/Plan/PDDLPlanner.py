import collections
import datetime
import os
import re
import subprocess

import Configuration
from Utils import Logger


class PDDLPlanner:

    def __init__(self):
        pass

    def pddl_plan(self):
        pddl_plan = None
        if Configuration.PLANNER == Configuration.FD_PLANNER:
            pddl_plan = self.FD()
        elif Configuration.PLANNER == Configuration.FF_PLANNER:
            pddl_plan = self.FF()

        # Add final stop action
        if pddl_plan is not None:
            pddl_plan.append("STOP()")

        return pddl_plan


    def FD(self):
        """
        Compute the plan using FastDownward planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
        into the "PDDL" folder. The plan is written in the output file "sas_plan"
        :return: a plan
        """

        # DEBUG
        start = datetime.datetime.now()

        # HEURISTIC SEARCH
        # bash_command = "./PAL/Plan/PDDL/Planners/FD/fast-downward.py --overall-time-limit {}" \
        #                " PAL/Plan/PDDL/domain.pddl PAL/Plan/PDDL/facts.pddl " \
        #                "--evaluator \"hff=ff()\" --evaluator \"hcea=cea()\" " \
        #                "--search \"lazy_greedy([hff,hcea],preferred=[hff,hcea])\"".format(Configuration.PLANNER_TIMELIMIT)

        # OPTIMAL SEARCH
        bash_command = "./PAL/Plan/PDDL/Planners/FD/fast-downward.py --overall-time-limit {}" \
                       " PAL/Plan/PDDL/domain.pddl PAL/Plan/PDDL/facts.pddl " \
                       "--search \"astar(hmax())\"".format(Configuration.PLANNER_TIMELIMIT)

        # ./Planners/FD/fast-downward.py PDDL/domain.pddl PDDL/facts.pddl --evaluator "hff=ff()" --evaluator "hcea=cea()" --search "lazy_greedy([hff,hcea],preferred=[hff,hcea])"

        # ./PDDL/Planners/FD/fast-downward.py PDDL/domain.pddl PDDL/facts.pddl --search "astar(hmax())"
        # ./PDDL/Planners/FD/fast-downward.py PDDL/domain.pddl PDDL/facts.pddl --search "astar(lmcut())" teoricamente piu lento di hmax

        process = subprocess.Popen(bash_command.replace("\"", "").split(), stdout=subprocess.PIPE)

        output, error = process.communicate()

        # DEBUG
        end = datetime.datetime.now()
        print("FD computational time: {}".format(end - start))

        syntax_plan = []

        if str(output).find("no solution") != -1:
            return None

        if str(output).lower().find("time limit has been reached") != -1 \
                or str(output).lower().find("translate exit code: 21") != -1 \
                or str(output).lower().find("translate exit code: -9") != -1:
            # translate exit code: 21 means time limit reached in the translator
            return None

        with open(os.path.join(os.getcwd(), 'sas_plan'), 'r') as file:

            data = file.read().split('\n')

            data = list(filter(lambda row: row.find(";") == -1 and len(row) > 0, data))

            # Check plan exists
            # if len(data) == 0:
            #     return [], False

            for el in data:
                el = el[1:-1]
                params = el.split()
                # tmp = params[0].replace("-","_") + "("
                tmp = params[0] + "("

                tmp += ",".join(params[1:])

                tmp += ")"

                syntax_plan.append(tmp.upper())

        os.remove(os.path.join(os.getcwd(), 'sas_plan'))

        return syntax_plan


    def FF(self):
        """
        Compute the plan using FastForward planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
        into the "PDDL" folder.
        :return: a plan
        """

        # DEBUG
        start = datetime.datetime.now()

        # Check empty state
        with open("./PAL/Plan/PDDL/facts.pddl", "r") as f:
            pddl_state = [el.strip() for el in f.read().split("\n") if el.strip() != '']

            problem_objs = [el for el in re.findall(":objects(.*?)\)", "++".join(pddl_state))[0].split("++") if
                        el.strip() != ""]
        if len(problem_objs) == 0:
            return None


        bash_command = "./PAL/Plan/PDDL/Planners/FF/ff -o PAL/Plan/PDDL/domain.pddl -f PAL/Plan/PDDL/facts.pddl"

        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        output, error = process.communicate()

        begin_index = -1
        end_index = -1
        result = str(output).split("\\n")

        found_plan = False
        for el in result:
            if el.__contains__("found legal plan as follows"):
                found_plan = True
                break

            elif el.__contains__("empty plan solves it"):
                # # If there is at least one object type not yet discovered, explore the environment
                # for row in result:
                #     if row.__contains__("unknown or empty type"):
                #         return None
                found_plan = True
                break

        if not found_plan:
            return None

        # for el in result:
        #     if el.__contains__("No plan will solve it"):
        #         return None
        #     elif el.__contains__("unknown or empty type"):
        #         return None
        #     elif el.__contains__("unknown constant"):
        #         Logger.write('WARNING: unknown constant in pddl problem file. Cannot compute any plan.')
        #         return None

        for i in range(len(result)):
            if not result[i].find("step"):
                begin_index = i

            elif not result[i].find("time"):
                end_index = i-2

        plan = [result[i].split(":")[1].replace("\\r", "") for i in range(begin_index, end_index)
                if result[i].split(":")[1].replace("\\r", "").lower().strip() != 'reach-goal']
        syntax_plan = []

        if len(plan) == 0:

            # If goal has been reached with discovered objects
            for el in result:
                if el.__contains__("empty plan solves it") or el.__contains__("found legal plan"):
                    # If there is at least one object type not yet discovered, explore the environment
                    for row in result:
                        if row.__contains__("unknown or empty type"):
                            return None
            return syntax_plan

        # DEBUG
        end = datetime.datetime.now()
        # print("FF computational time: {}".format(end-start))

        for el in plan:
            tmp = el
            tmp = re.sub("[ ]", ",", tmp.strip())
            tmp = tmp.replace(",", "(", 1)
            tmp = tmp + ")"
            tmp = tmp[:tmp.index('(')].replace("-", "_") + tmp[tmp.index('('):]
            syntax_plan.append(tmp)


        return syntax_plan
