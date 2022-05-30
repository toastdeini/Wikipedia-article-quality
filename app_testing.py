# Imports
import streamlit as st
import pickle
from src.parse_it import *

f = open('models/xgb_model.sav', 'rb')
model = pickle.load(f)
f.close()

decoder = {
    0 : 'Looking good!',
    1 : 'Hm, this could use some work...'
}

st.title("Is this Wikipedia article up to snuff?")

article = st.text_area(
"Find out by pasting the text of your article below!",
value="The Papuan mountain pigeon (Gymnophaps albertisii) is a species of bird\
 in the pigeon family Columbidae. It is found in the Bacan Islands, New Guinea,\
 the D'Entrecasteaux Islands, and the Bismarck Archipelago, where it inhabits\
 primary forest, montane forest, and lowlands. It is a medium-sized species of\
 pigeon, being 33–36 cm (13–14 in) long and weighing 259 g (9.1 oz) on average.\
 Adult males have slate-grey upperparts, chestnut-maroon throats and bellies,\
 whitish breasts, and a pale grey terminal tail band. The lores and orbital region\
 are bright red. Females are similar, but have grayish breasts and grey edges to\
 the throat feathers.\n\nThe Papuan mountain pigeon is frugivorous, feeding on\
 figs and drupes. It breeds from October to March in the Schrader Range, but\
 may breed throughout the year across its range. It builds nests out of sticks\
 and twigs in a tree or makes a ground nest in short dry grass, and lays a single egg.\
 The species is very social and is usually seen in flocks of 10–40 birds, although\
 some groups can have as many as 80 individuals. It is listed as being of least concern\
 by the International Union for Conservation of Nature (IUCN) on the IUCN Red List due\
 to its large range and lack of significant population decline.\n\nThe Papuan mountain\
 pigeon was described as Gymnophaps albertisii by the Italian zoologist Tommaso Salvadori\
 in 1874 on the basis of specimens from Andai, New Guinea. It is the type species of the\
 genus Gymnophaps, which was created for it.[3] The generic name is derived from the\
 Ancient Greek words γυμνος (gumnos), meaning bare, and φαψ (phaps), meaning pigeon.\
 The specific name albertisii is in honour of Luigi D'Albertis, an Italian botanist and\
 zoologist who worked in the East Indies and New Guinea.[4] Papuan mountain pigeon is\
 the official common name designated by the International Ornithologists' Union.\
 Other common names for the species include mountain pigeon (which is also used for\
 Gymnophaps pigeons in general), bare-eyed mountain pigeon, bare-eyed pigeon (which is\
 also used for Patagioenas corensis), and D'Albertis's mountain pigeon.\n\nThe Papuan\
 mountain pigeon is one of four species in the mountain pigeon genus Gymnophaps in the pigeon\
 family Columbidae, which is found in Melanesia and the Maluku Islands. It forms a\
 superspecies with the other species in its genus. Within its family, the genus\
 Gymnophaps is sister to Lopholaimus, and these two together form a clade sister to\
 Hemiphaga.[8] The Papuan mountain pigeon has two subspecies:[a][3]",
height=200,
max_chars=100000,
placeholder='Enter some text!'
)

pred = model.predict([parse_doc(article)])

st.write(decoder[pred[0]])