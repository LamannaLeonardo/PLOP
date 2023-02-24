from PAL.Plan.EventPlanner import EventPlanner
from PAL.Plan.PDDLPlanner import PDDLPlanner
from Utils import Logger


class MetaPlanner:

    def __init__(self, knowledge_base):

        # Set knowledge base
        self.knowledge_base = knowledge_base

        # Set pddl planner
        self.pddl_planner = PDDLPlanner()

        # Set low level action planner
        self.event_planner = EventPlanner(self.knowledge_base)

        # Set pddl plan
        self.pddl_plan = None


    def plan(self):

        action = None

        while action is None:

            # Compute a pddl plan
            if self.pddl_plan is None:
                self.pddl_plan = self.pddl_planner.pddl_plan()

            # If no pddl plan can be computed explore the environment to learn new objects and properties
            if self.pddl_plan is None:
                action = self.event_planner.explore()

            # Set first pddl plan action as an event planner subgoal
            else:
                # Check if previous event planner subgoal has been reached, and update the subgoal with the next one
                if self.event_planner.subgoal is None:
                    self.event_planner.subgoal = self.pddl_plan.pop(0)
                    Logger.write('Changing event planner subgoal to: {}'.format(self.event_planner.subgoal))

                    # Invalidate the previous plan in the event planner
                    self.event_planner.event_plan = None
                    self.event_planner.action_compiler.explored_goal_cells = []

                # Plan to achieve the subgoal with the event planner
                action = self.event_planner.event_planning()

                # If the previous subgoal has been reached, update the subgoal with the next one
                while self.event_planner.subgoal is None:
                    self.event_planner.subgoal = self.pddl_plan.pop(0)
                    Logger.write('Changing event planner subgoal to: {}'.format(self.event_planner.subgoal))

                    # Invalidate the previous plan in the event planner
                    self.event_planner.event_plan = None
                    self.event_planner.action_compiler.explored_goal_cells = []

                    # Plan to achieve the subgoal with the event planner
                    action = self.event_planner.event_planning()

                # If the current subgoal is not reachable, invalidate the plan
                if action is None:
                    self.pddl_plan = None
                    self.event_planner.subgoal = None

        return action
