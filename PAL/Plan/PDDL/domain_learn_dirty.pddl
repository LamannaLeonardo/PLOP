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
		(dirty ?o - object)
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
        (supervised_dirty_plate )
        (supervised_dirty_mug )
        (supervised_dirty_cup )
        (supervised_dirty_pot )
        (supervised_dirty_bowl )
        (supervised_dirty_pan )
        (supervised_dirty_bed )
        (supervised_dirty_cloth )
        (supervised_dirty_mirror )
        (supervised_notdirty_plate )
        (supervised_notdirty_mug )
        (supervised_notdirty_cup)
        (supervised_notdirty_pot )
        (supervised_notdirty_bowl )
        (supervised_notdirty_pan )
        (supervised_notdirty_bed )
        (supervised_notdirty_cloth )
        (supervised_notdirty_mirror )
)


(:action dirty
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (dirtyable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (dirty ?param_1)
		         )
)


(:action clean
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (dirtyable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		                    (explored)
		              )
		:effect
		        (and
                    (manipulated ?param_1)
		            (not (dirty ?param_1))
		         )
)


(:action get_close_and_look_at_dirtyable
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (dirtyable ?param_1)
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


(:action scan_dirty
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (dirty ?param_1)
		                (manipulated ?param_1)
                        (explored)
		              )
		:effect
		        (and
		            (scanned ?param_1)
		         )
)


(:action scan_clean
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (inspected ?param_1)
		                (not (dirty ?param_1))
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

