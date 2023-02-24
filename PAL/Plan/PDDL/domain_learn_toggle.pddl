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
		(toggled ?o - object)
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
        (supervised_toggled_laptop )
        (supervised_toggled_microwave )
        (supervised_toggled_desklamp )
        (supervised_toggled_candle )
        (supervised_toggled_cellphone )
        (supervised_toggled_faucet )
        (supervised_toggled_showerhead )
        (supervised_toggled_coffeemachine )
        (supervised_toggled_floorlamp )
        (supervised_toggled_desktop )
        (supervised_toggled_toaster )
        (supervised_toggled_television )
        (supervised_nottoggled_laptop )
        (supervised_nottoggled_microwave )
        (supervised_nottoggled_desklamp )
        (supervised_nottoggled_candle )
        (supervised_nottoggled_cellphone )
        (supervised_nottoggled_faucet )
        (supervised_nottoggled_toaster )
        (supervised_nottoggled_showerhead )
        (supervised_nottoggled_coffeemachine )
        (supervised_nottoggled_floorlamp )
        (supervised_nottoggled_desktop )
        (supervised_nottoggled_television )
)


(:action toggle_on
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (toggleable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (toggled ?param_1)
		         )
)


(:action toggle_off
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (toggleable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (not (toggled ?param_1))
		         )
)


(:action get_close_and_look_at_toggleable
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (toggleable ?param_1)
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


(:action scan_toggled_on
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (toggled ?param_1)
		                (manipulated ?param_1)
                        (explored)
		              )
		:effect
		        (and
		            (scanned ?param_1)
		         )
)


(:action scan_toggled_off
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (not (toggled ?param_1))
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

