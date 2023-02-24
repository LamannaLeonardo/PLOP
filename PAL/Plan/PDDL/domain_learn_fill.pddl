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
		(filled ?o - object)
		(discovered ?o - object)
		(openable ?o - object)
		(open ?o - object)
		(dirtyable ?o - object)
		(fillable ?o - object)
		(toggleable ?o - object)
		(pickupable ?o - object)
		(receptacle ?o - object)
		(viewing ?o - object)
		(inspected ?o - object)
		(manipulated ?o - object)
		(scanned ?o - object)
        (explored)
        (supervised_filled_kettle )
        (supervised_filled_winebottle )
        (supervised_filled_bottle )
        (supervised_filled_cup )
        (supervised_filled_pot )
        (supervised_filled_wateringcan )
        (supervised_filled_bowl )
        (supervised_filled_houseplant )
        (supervised_filled_mug )
        (supervised_notfilled_kettle )
        (supervised_notfilled_winebottle )
        (supervised_notfilled_bottle )
        (supervised_notfilled_cup )
        (supervised_notfilled_pot )
        (supervised_notfilled_wateringcan)
        (supervised_notfilled_bowl )
        (supervised_notfilled_houseplant )
        (supervised_notfilled_mug )
)


(:action fill
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (fillable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (filled ?param_1)
		         )
)


(:action unfill
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (fillable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (not (filled ?param_1))
		         )
)


(:action get_close_and_look_at_fillable
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (fillable ?param_1)
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


(:action scan_filled
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (filled ?param_1)
		                (manipulated ?param_1)
                        (explored)
		              )
		:effect
		        (and
		            (scanned ?param_1)
		         )
)


(:action scan_unfilled
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (not (filled ?param_1))
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

