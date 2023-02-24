(define (problem ithor-appleinbox)
(:domain ithor)
(:objects
laptop_0 - laptop
cabinet_0 - cabinet
box_0 - box
)
(:init
(close_to cabinet_0)
(close_to cabinet_0)
(discovered box_0)
(discovered cabinet_0)
(discovered laptop_0)
(explored )
(hand_free )
(inspected cabinet_0)
(open cabinet_0)
(openable box_0)
(openable cabinet_0)
(openable laptop_0)
(pickupable box_0)
(pickupable laptop_0)
(receptacle box_0)
(receptacle cabinet_0)
(supervised_opened_cabinet )
(toggleable laptop_0)
)
(:goal
(and (or (and (supervised_opened_book ) (supervised_notopened_book )) (forall (?o1 - book) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_box ) (supervised_notopened_box )) (forall (?o1 - box) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_cabinet ) (supervised_notopened_cabinet )) (forall (?o1 - cabinet) (and (not (open ?o1)) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_drawer ) (supervised_notopened_drawer )) (forall (?o1 - drawer) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_fridge ) (supervised_notopened_fridge )) (forall (?o1 - fridge) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_kettle ) (supervised_notopened_kettle )) (forall (?o1 - kettle) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_laptop ) (supervised_notopened_laptop )) (forall (?o1 - laptop) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_microwave ) (supervised_notopened_microwave )) (forall (?o1 - microwave) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_safe ) (supervised_notopened_safe )) (forall (?o1 - safe) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_showercurtain ) (supervised_notopened_showercurtain )) (forall (?o1 - showercurtain) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_showerdoor ) (supervised_notopened_showerdoor )) (forall (?o1 - showerdoor) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))) (or (and (supervised_opened_toilet ) (supervised_notopened_toilet )) (forall (?o1 - toilet) (and (open ?o1) (manipulated ?o1) (scanned ?o1)))))
))