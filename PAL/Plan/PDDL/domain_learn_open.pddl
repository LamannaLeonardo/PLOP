(define (domain ithor)
(:requirements :typing)
(:types
alarmclock
aluminumfoil
apple
armchair
baseballbat
basketball
bathtub
bathtubbasin
bed
blinds
book
boots
bottle
bowl
box
bread
butterknife
cd
cabinet
candle
cellphone
chair
cloth
coffeemachine
coffeetable
countertop
creditcard
cup
curtains
desk
desklamp
desktop
diningtable
dishsponge
dogbed
drawer
dresser
dumbbell
egg
faucet
floor
floorlamp
footstool
fork
fridge
garbagebag
garbagecan
handtowel
handtowelholder
houseplant
kettle
keychain
knife
ladle
laptop
laundryhamper
lettuce
lightswitch
microwave
mirror
mug
newspaper
ottoman
painting
pan
papertowelroll
pen
pencil
peppershaker
pillow
plate
plunger
poster
pot
potato
remotecontrol
roomdecor
safe
saltshaker
scrubbrush
shelf
shelvingunit
showercurtain
showerdoor
showerglass
showerhead
sidetable
sink
sinkbasin
soapbar
soapbottle
sofa
spatula
spoon
spraybottle
statue
stool
stoveburner
stoveknob
tvstand
tabletopdecor
teddybear
television
tennisracket
tissuebox
toaster
toilet
toiletpaper
toiletpaperhanger
tomato
towel
towelholder
vacuumcleaner
vase
watch
wateringcan
window
winebottle - object
)

(:predicates
        (hand_free)
		(holding ?o - object)
		(on ?o1 ?o2 - object)
		(close_to ?o - object)
		(open ?o - object)
		(discovered ?o - object)
		(dirtyable ?o - object)
		(openable ?o - object)
		(fillable ?o - object)
		(toggleable ?o - object)
		(pickupable ?o - object)
		(receptacle ?o - object)
		(viewing ?o - object)
		(inspected ?o - object)
		(manipulated ?o - object)
		(scanned ?o - object)
        (explored)
        (supervised_opened_blinds )
        (supervised_opened_book )
        (supervised_opened_box )
        (supervised_opened_cabinet )
        (supervised_opened_drawer )
        (supervised_opened_fridge )
        (supervised_opened_kettle )
        (supervised_opened_laptop )
        (supervised_opened_microwave )
        (supervised_opened_safe )
        (supervised_opened_showercurtain )
        (supervised_opened_showerdoor )
        (supervised_opened_toilet )
        (supervised_notopened_blinds )
        (supervised_notopened_book )
        (supervised_notopened_box )
        (supervised_notopened_cabinet )
        (supervised_notopened_drawer )
        (supervised_notopened_fridge )
        (supervised_notopened_kettle )
        (supervised_notopened_laptop )
        (supervised_notopened_microwave )
        (supervised_notopened_safe )
        (supervised_notopened_showercurtain )
        (supervised_notopened_showerdoor )
        (supervised_notopened_toilet )
)


(:action open
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (openable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (open ?param_1)
		         )
)


(:action close
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (openable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (not (open ?param_1))
		         )
)


(:action get_close_and_look_at_openable
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (openable ?param_1)
		                (inspected ?param_1)
		                (or
		                (not (close_to ?param_1))
		                (not (viewing ?param_1))
		                )
                        (explored)
		              )
		:effect
		        (and
		            (close_to ?param_1)
		            (viewing ?param_1)
		            (forall (?x - object) (not (viewing ?x)))
		         )
)


(:action inspect
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (not (inspected ?param_1))
                        (explored)
		              )
		:effect
		        (and
		            (inspected ?param_1)
		            (close_to ?param_1)
		            (viewing ?param_1)
		            (forall (?x - object) (not (viewing ?x)))
		         )
)


(:action scan_opened
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (open ?param_1)
		                (manipulated ?param_1)
                        (explored)
		              )
		:effect
		        (and
		            (scanned ?param_1)
		         )
)


(:action scan_closed
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (not (open ?param_1))
		                (manipulated ?param_1)
                        (explored)
		              )
		:effect
		        (and
		            (scanned ?param_1)
		         )
)


(:action stop
		:parameters ()
		:precondition (and)
		:effect (and)
)


)

