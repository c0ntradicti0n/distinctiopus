import doctest

negation_list = ["no", "not", "none", "never", "nothing", "nobody", "nowhere", "neither", "nor", "non", "n't", "cannot", "prevent", 'disclaim', 'disclaims']
conjunction_list = ["and", "but"]
disjunction_list = ["or"]
conditional_list = ['if', 'when', 'which', 'who', 'that']
all_quantors = ['all', 'every', 'everybody', 'everytime', 'everything', 'anything', 'any']
existential_quantors = ['some', 'certain', 'few']

logic_dict = {
    '~': negation_list,
    #'|': disjunction_list + conjunction_list + conditional_list,
    '|': disjunction_list,
    '&': conjunction_list + conditional_list,
#    '>>': conditional_list,
#    'FA': all_quantors,
#    'TE': existential_quantors
}

counterpart_words = []

def test_antonymdict():
    ''' Check if the antonymdict hast correct shape

        >>> all([isinstance(x, str) or isinstance(x, tuple)
        ... for x in antonym_dict['lemma_'].keys()])
        True

        >>> all([isinstance(x, list)
        ...     for x in antonym_dict['lemma_'].values()])
        True
        >>> all([isinstance(x, str) or isinstance(x, tuple)
        ...      for lists in antonym_dict['lemma_'].values()
        ...      for x in lists])
        True
    '''
    pass

antonym_dict = {'lemma_': {'part': ['wholly'],  'equivocally': ['unambiguously'], 'have': ['lack'],
'common': ['individual', 'single', 'derive'], 'correspond': ['disagree'], 'differ': ['common', 'derive'], 'real':    ['unreal',
'nominal', 'insubstantial'], 'man': ['civilian', 'woman','baby'], 'figure': ['ground'], 'can': ['hire'], 'lie': ['sitting',
'sit', 'stand'], 'claim': ['forfeit', 'disclaim'], 'will': ['disinherit'], 'appropriate': ['inappropriate'], 'on':
['off'], 'other': ['same'], 'answer': ['question'], 'same': ['other', 'different'], 'identical': ['fraternal'],
'some': ['all', 'no'], 'courageous': ['cowardly'], 'courage': ['cowardice'], 'simple': ['complex', 'compound', 'composite'],
'latter': ['former'], 'expression': ['misconstruction'], 'former': ['latter'], 'run': ['malfunction', 'idle'],
'wins': ['losings', 'lose', 'fail', 'fall_back'], 'present': ['past', 'future', 'absent'],
'individual': ['common'], 'do': ['unmake'], 'whole': ['half'], 'incapable': ['capable'], 'existence': ['nonbeing',
'nonexistence'], 'certain': ['unsealed', 'uncertain'], 'point': ['unpointedness'], 'grammatical': ['ungrammatical'],
'mind': ['forget'], 'whiteness': ['blacken', 'black'], 'human': ['nonhuman'], 'there': ['here'], 'much': ['little',
'slight'], 'generally': ['narrowly', 'specifically'], 'prevent': ['allow', 'let'], 'all': ['some', 'no'],
'different': ['same'], 'kind': ['unkind'], 'take': ['give'], 'winged': ['wingless'], 'aquatic': ['terrestrial'],
'subordinate': ['independent', 'insubordinate', 'dominant'], 'less': ['more'], 'place': ['divest'],
'action': ['inaction'], 'white': ['black', 'blackness'], 'double': ['single'], 'half': ['whole'], 'greater':
['lesser'], 'fall': ['rise'], 'sitting': ['lie', 'standing', 'arise', 'stand'], 'indicate': ['contraindicate'],
'shod': ['unshod', 'discalced'], 'armed': ['unarmed', 'armless', 'disarm'], 'involve': ['obviate'], 'affirmation':
['reversal'], 'positive': ['negative'], 'negative': ['positive', 'affirmative'], 'arise': ['lie', 'sitting', 'sit'],
'admit': ['exclude', 'deny'], 'true': ['false'], 'false': ['true', 'TRUE'], 'TRUE': ['untruthful', 'false'],
'primary': ['secondary'], 'definite': ['indefinite'], 'secondary': ['primary'], 'call': ['put_option'], 'within':
['outside'], 'include': ['except', 'exclude'], 'plain': ['patterned', 'cheer', 'fancy'], 'apply': ['defy',
'exempt'], 'regard': ['disrespect', 'inattentiveness', 'disesteem'], 'colour': ['black-and-white', 'colorlessness',
'discolor'], 'except': ['include'], 'particular': ['general', 'universal'], 'last': ['first'], 'impossible':
['possible'], 'truly': ['insincerely'], 'relate': ['decouple'], 'instructive': ['uninstructive'], 'proper':
['improper'], 'give': ['take'], 'general': ['particular', 'specific'], 'properly': ['improperly'], 'virtue':
['demerit'], 'good': ['bad', 'badness', 'evil'], 'remain': ['change'], 'exclude': ['admit', 'include'], 'convey':
['take_away'], 'appropriately': ['inappropriately'], 'make': ['unmake', 'break'], 'exact': ['inexact'],
'irrelevant': ['relevant'], 'just': ['unjust'], 'hold': ['let_go_of', 'disagree'], 'characteristic':
['uncharacteristic'], 'clear': ['convict', 'cloudy', 'opaque', 'unclear', 'bounce', 'ill-defined', 'overcast',
'clutter'], 'follow': ['predate', 'precede'], 'terrestrial': ['aquatic'], 'appear': ['disappear'], 'mark':
['ignore'], 'above': ['below'], 'univocal': ['equivocal'], 'establish': ['disprove', 'abolish'], 'here': ['there'],
'single': ['double'], 'like': ['different', 'unlike'], 'far': ['near'], 'differentiate': ['dedifferentiate',
'integrate'], 'determinate': ['indeterminate'], 'cover': ['uncover'], 'large': ['little', 'small'], 'wide':
['narrow'], 'extension': ['flexion'], 'many': ['few'], 'little': ['much'], 'small': ['large'], 'quantitative':
['qualitative'], 'vary': ['conform'], 'beautiful': ['ugly'], 'warm': ['cool'], 'capable': ['incapable'], 'find':
['lose'], 'unable': ['able'], 'bring': ['take_away'], 'forward': ['reverse'], 'black': ['whiteness', 'white'],
'bad': ['good', 'goodness'], 'cold': ['hot'], 'capacity': ['incapacity'], 'agree': ['differ'], 'question':
['answer'], 'rise': ['fall', 'set'], 'think': ['forget'], 'sit': ['lie', 'stand'], 'still': ['agitate', 'moving',
'no_longer', 'louden', 'sparkling'], 'allow': ['deny'], 'difference': ['sameness'], 'change': ['remain', 'rest'],
'hot': ['cold'], 'enter': ['leave'], 'unaltered': ['altered'], 'respect': ['disrespect', 'disesteem'], 'come':
['go'], 'accord': ['disagreement'], 'contention': ['cooperation'], 'unsound': ['sound'], 'truth': ['falsity'],
'falsity': ['truth'], 'power': ['powerlessness', 'inability'], 'short': ['long'], 'health': ['illness'],
'blackness': ['white'], 'up': ['downwards'], 'let': ['prevent'], 'continuous': ['discontinuous'], 'relative':
['absolute'], 'surface': ['subsurface', 'overhead'], 'solid': ['hollow', 'liquid', 'gaseous'], 'join': ['disjoin'],
'separate': ['unite', 'joint'], 'generalize': ['specify'], 'ever': ['never'], 'possible': ['impossible'], 'always':
['never'], 'long': ['short'], 'vocal': ['instrumental'], 'distinct': ['indistinct'], 'rest': ['move'], 'past':
['present', 'future'], 'future': ['present', 'past'], 'bear': ['bull'], 'show': ['disprove', 'hide'], 'well':
['sick', 'ill'], 'order': ['disorder', 'deregulate', 'disorderliness'], 'naturally': ['artificially',
'unnaturally'], 'right': ['wrong'], 'intrinsic': ['extrinsic'], 'obvious': ['unobvious'], 'act': ['refrain'],
'external': ['internal'], 'standard': ['nonstandard'], 'few': ['many'], 'happen': ['dematerialize'], 'moment':
['inconsequence'], 'sick': ['keep_down', 'well'], 'healthy': ['unhealthy'], 'qualify': ['disqualify'], 'even':
['odd'], 'plausibly': ['incredibly'], 'below': ['above'], 'spatial': ['nonspatial'], 'set': ['rise'], 'equality':
['inequality'], 'inequality': ['equality'], 'equal': ['differ', 'unequal'], 'unequal': ['equal'], 'similarity':
['dissimilarity'], 'superior': ['adscript', 'inferior', 'subscript'], 'superiority': ['inferiority'],
'significance': ['insignificance'], 'son': ['daughter'], 'similar': ['unlike', 'dissimilar', 'unalike'], 'note':
['ignore'], 'stand': ['lie', 'sitting', 'sit'], 'unlike': ['same', 'like', 'similar'], 'more': ['less'], 'express':
['local'], 'knowable': ['unknowable'], 'perceptible': ['imperceptible'], 'accurately': ['inaccurately'],
'necessarily': ['unnecessarily'], 'connexion': ['unconnectedness'], 'reciprocal': ['nonreciprocal'], 'original':
['unoriginal'], 'inaccurate': ['accurate'], 'accurate': ['inaccurate'], 'necessary': ['inessential', 'unnecessary'],
'adequately': ['inadequately'], 'connect': ['disconnect', 'decouple', 'unplug'], 'head': ['foot'], 'new': ['old'],
'add': ['subtract', 'take_away'], 'clearly': ['unintelligibly'], 'acknowledge': ['deny'], 'biped': ['quadrupedal'],
'correct': ['wrong'], 'leave': ['come'], 'correctly': ['wrong'], 'disappear': ['appear'], 'withdraw': ['advance',
'progress'], 'cease': ['begin', 'continue'], 'essential': ['inessential', 'adjective'], 'exactly': ['imprecisely'],
'easy': ['difficult'], 'construct': ['misconception'], 'necessitate': ['obviate'], 'usually': ['unusually'],
'acquire': ['lose'], 'difficult': ['easy'], 'begin': ['cease'], 'know': ['ignore'], 'equally': ['unevenly'],
'square': ['circular'], 'heat': ['cool'], 'sweetness': ['unpleasantness'], 'fire': ['hire'], 'out': ['safe'],
'possibility': ['impossibility'], 'outside': ['within'], 'complete': ['incomplete'], 'prove': ['disprove'], 'raise':
['descent', 'lower', 'demote', 'level'], 'advantage': ['penalty', 'disadvantage'], 'moderate': ['immoderate'],
'displace': ['hire'], 'mental': ['physical'], 'justice': ['injustice'], 'dislodge': ['lodge'], 'dismiss': ['hire'],
'quickly': ['easy'], 'opposite': ['synonym', 'alternate'], 'dispose': ['disqualify', 'indispose'], 'ill': ['good',
'well'], 'go': ['come'], 'incline': ['indispose'], 'permanent': ['impermanent'], 'type': ['antitype'], 'retentive':
['short'], 'volatile': ['nonvolatile'], 'wrong': ['right', 'correct'], 'specific': ['general'], 'incapacity':
['capacity'], 'ease': ['difficulty'], 'avoid': ['validate', 'confront'], 'defeat': ['victory'], 'unhealthy':
['healthy'], 'ordinarily': ['unusually'], 'lack': ['have'], 'softness': ['hardness'], 'hardness': ['softness'],
'enable': ['disable'], 'withstand': ['surrender'], 'presence': ['absence'], 'sweet': ['sour', 'dry', 'salty'],
'ashamed': ['unashamed'], 'afraid': ['unafraid'], 'probable': ['improbable'], 'shame': ['honor'], 'natural':
['supernatural', 'flat', 'sharp', 'artificial', 'unnatural'], 'ineffective': ['effective'], 'speedily': ['slowly'],
'constitutional': ['unconstitutional'], 'fear': ['fearlessness'], 'constitutionally': ['unconstitutionally'],
'insanity': ['sanity'], 'abnormal': ['normal'], 'altogether': ['partially'], 'vex': ['reassure'], 'lose': ['wins',
'find', 'acquire', 'regain'], 'somejupyter notebookwhat': ['unreasonably'], 'straightness': ['indirectness', 'crookedness',
'curliness'], 'straight': ['curly', 'indirectly', 'curved', 'coiled', 'crooked'], 'curve': ['straight_line'],
'density': ['distribution'], 'roughness': ['smoothness'], 'smoothness': ['roughness'], 'smooth': ['rough'],
'evenly': ['unevenly'], 'rough': ['smooth'], 'dependent': ['independent'], 'adjective': ['essential'],
'consideration': ['inconsideration'], 'boxing': ['unbox'], 'upright': ['unerect'], 'often': ['rarely',
'infrequently'], 'injustice': ['justice'], 'unjust': ['just'], 'red': ['gain'], 'grant': ['deny'], 'difficulty':
['ease'], 'circular': ['square'], 'likeness': ['unlikeness'], 'unlikeness': ['likeness'], 'feature': ['miss'],
'extraordinary': ['ordinary'], 'heating': ['cool'], 'cool': ['warm', 'heat', 'heating'], 'glad': ['sad'],
'intelligible': ['unintelligible'], 'affirmative': ['negative'], 'intermediate': ['terminal'], 'necessity':
['inessential'], 'odd': ['even'], 'badness': ['good'], 'goodness': ['evil', 'bad'], 'universal': ['particular'],
'suffer': ['enjoy', 'be_well'], 'toothless': ['toothed'], 'blind': ['sighted'], 'birth': ['death'], 'affirm':
['negate'], 'deny': ['admit', 'allow', 'acknowledge', 'grant'], 'denial': ['prosecution'], 'need': ['obviate'],
'receptive': ['unreceptive'], 'member': ['nonmember'], 'advance': ['withdraw'], 'indeterminate': ['determinate'],
'absent': ['present'], 'slight': ['much'], 'improve': ['worsen'], 'completely': ['partially'], 'progress':
['retreat', 'withdraw', 'regress'], 'move': ['refrain', 'rest', 'stay_in_place', 'stand_still', 'stay'],
'improvement': ['decline'], 'regain': ['lose'], 'course': ['unnaturally'], 'able': ['unable'], 'nonexistent':
['existent'], 'evil': ['good', 'goodness'], 'cowardice': ['courage'], 'turn': ['unbend'], 'possibly':
['impossibly'], 'require': ['obviate'], 'actual': ['possible'], 'primarily': ['secondarily'], 'old': ['new'],
'reverse': ['forward'], 'directly': ['indirectly'], 'posterior': ['anterior'], 'honourable': ['dishonorable'],
'priority': ['posteriority'], 'honour': ['disrespect', 'dishonor'], 'love': ['hate'], 'reasonably':
['unreasonably'], 'divide': ['multiply', 'unite'], 'unqualified': ['competent', 'qualified'], 'increase': ['fall',
'diminution'], 'diminution': ['increase', 'augmentation'], 'motion': ['motionlessness'], 'addition':
['subtraction'], 'upwards': ['down'], 'downwards': ['up'], 'qualitative': ['quantitative'], 'piece':
['disassemble'], 'apparel': ['undress'], 'ring': ['open_chain'], 'foot': ['head'], 'content': ['discontented',
'discontent'], 'wife': ['husband'], 'husband': ['wife'], 'remote': ['close'], 'live': ['dead', 'recorded'],
'ordinary': ['extraordinary'], 'wholly': ['part'], 'unambiguously': ['equivocally'], 'abstain': ['have', 'take'],
'refuse': ['have', 'take'], 'miss':
                               ['have', 'feature'], 'uncommon': ['common'], 'disagree': ['correspond', 'hold',
'agree'], 'unreal': ['real'], 'nominal': ['real'], 'insubstantial': ['real'], 'civilian': ['man'], 'woman': ['man'],
'ground': ['figure'], 'hire': ['can', 'fire', 'displace', 'dismiss'], 'forfeit': ['claim'], 'disclaim': ['claim',
'take'], 'disinherit': ['will', 'leave'], 'inappropriate': ['appropriate'], 'off': ['on'], 'fraternal':
['identical'], 'cowardly': ['courageous'], 'complex': ['simple'], 'compound': ['simple'], 'misconstruction':
['expression'], 'malfunction': ['run', 'go'], 'idle': ['run'], 'losings': ['wins'], 'fail': ['wins'], 'fall_back':
['wins', 'advance'], 'unmake': ['do', 'make'], 'fractional': ['whole'], 'partially': ['whole', 'all', 'altogether',
'completely'], 'nonbeing': ['existence'], 'nonexistence': ['existence'], 'unsealed': ['certain'], 'uncertain':
['certain'], 'unpointedness': ['point'], 'ungrammatical': ['grammatical'], 'forget': ['mind', 'think'], 'blacken':
['whiteness', 'white'], 'nonhuman': ['human'], 'narrowly': ['generally'], 'specifically': ['generally'], 'unkind':
['kind'], 'obviate': ['take', 'involve', 'necessitate', 'need', 'require'], 'wingless': ['winged'], 'amphibious':
['aquatic', 'terrestrial'], 'independent': ['subordinate', 'dependent'], 'insubordinate': ['subordinate'],
'dominant': ['subordinate'], 'yes': ['no'], 'divest': ['place'], 'inaction': ['action'], 'multivalent': ['double'],
'univalent': ['double'], 'lesser': ['greater'], 'ascent': ['fall'], 'ascend': ['fall'], 'standing': ['sitting'],
'contraindicate': ['indicate'], 'unshod': ['shod'], 'discalced': ['shod'], 'unarmed': ['armed'], 'armless':
['armed'], 'disarm': ['armed'], 'reversal': ['affirmation'], 'neutral': ['positive', 'negative'], 'descend':
['arise', 'rise'], 'sit_down': ['arise', 'rise'], 'go_to_bed': ['arise', 'rise'], 'lie_down': ['arise', 'rise'],
'reject': ['admit'], 'untruthful': ['true', 'TRUE'], 'indefinite': ['definite'], 'put_option': ['call'],
'patterned': ['plain'], 'cheer': ['plain'], 'fancy': ['plain'], 'defy': ['apply'], 'exempt': ['apply'],
'disrespect': ['regard', 'respect', 'honour'], 'inattentiveness': ['regard'], 'disesteem': ['regard', 'respect'],
'black-and-white': ['colour'], 'colorlessness': ['colour'], 'discolor': ['colour'], 'first': ['last'],
'insincerely': ['truly'], 'decouple': ['relate', 'connect'], 'uninstructive': ['instructive'], 'improper':
['proper'], 'starve': ['give'], 'local': ['general', 'express'], 'improperly': ['properly', 'right'], 'demerit':
['virtue'], 'take_away': ['convey', 'bring', 'add'], 'inappropriately': ['appropriately'], 'break': ['make'],
'inexact': ['exact'], 'relevant': ['irrelevant'], 'inequitable': ['just'], 'unfair': ['just'], 'let_go_of':
['hold'], 'uncharacteristic': ['characteristic'], 'convict': ['clear'], 'cloudy': ['clear'], 'opaque': ['clear'],
'unclear': ['clear'], 'bounce': ['clear'], 'ill-defined': ['clear'], 'overcast': ['clear'], 'clutter': ['clear'],
'predate': ['follow'], 'precede': ['follow'], 'vanish': ['appear'], 'ignore': ['mark', 'note', 'know'], 'equivocal':
['univocal'], 'disprove': ['establish', 'show', 'prove'], 'abolish': ['establish'], 'multiple': ['single'],
'married': ['single'], 'unalike': ['like', 'similar'], 'dislike': ['like'], 'near': ['far'], 'dedifferentiate':
['differentiate'], 'integrate': ['differentiate'], 'uncover': ['cover'], 'narrow': ['wide'], 'flexion':
['extension'], 'tall': ['little', 'short'], 'big': ['small'], 'accentual': ['quantitative'], 'syllabic':
['quantitative'], 'conform': ['vary'], 'ugly': ['beautiful'], 'backward': ['forward'], 'aft': ['forward'], 'back':
['forward', 'advance'], 'whiten': ['black'], 'unregretful': ['bad'], 'hotness': ['cold'], 'descent': ['rise',
'raise'], 'wane': ['rise'], 'agitate': ['still'], 'moving': ['still'], 'no_longer': ['still'], 'louden': ['still'],
'sparkling': ['still'], 'forbid': ['allow', 'let'], 'sameness': ['difference'], 'stay': ['change', 'move'], 'exit':
['enter'], 'drop_out': ['enter'], 'altered': ['unaltered'], 'disagreement': ['accord'], 'cooperation':
['contention'], 'sound': ['unsound'], 'inaccuracy': ['truth'], 'falsehood': ['truth'], 'powerlessness': ['power'],
'inability': ['power'], 'illness': ['health'], 'down': ['up', 'upwards'], 'discontinuous': ['continuous'],
'absolute': ['relative'], 'subsurface': ['surface'], 'overhead': ['surface'], 'hollow': ['solid'], 'liquid':
['solid'], 'gaseous': ['solid'], 'disjoin': ['join'], 'unite': ['separate', 'divide'], 'joint': ['separate'],
'specify': ['generalize'], 'unretentive': ['long', 'retentive'], 'instrumental': ['vocal'], 'indistinct':
['distinct'], 'be_active': ['rest'], 'bull': ['bear'], 'hide': ['show'], 'badly': ['well'], 'disorder': ['order'],
'deregulate': ['order'], 'disorderliness': ['order'], 'artificially': ['naturally'], 'unnaturally': ['naturally',
'course'], 'falsify': ['right', 'correct'], 'incorrect': ['right', 'correct'], 'incorrectly': ['right',
'correctly'], 'left': ['right'], 'center': ['right'], 'extrinsic': ['intrinsic'], 'unobvious': ['obvious'],
'refrain': ['act', 'move'], 'internal': ['external'], 'nonstandard': ['standard'], 'dematerialize': ['happen'],
'inconsequence': ['moment'], 'keep_down': ['sick'], 'disqualify': ['qualify', 'dispose'], 'uneven': ['even'],
'incredibly': ['plausibly'], 'upstairs': ['below'], 'nonspatial': ['spatial'], 'inadequate': ['equal'], 'adequate':
['unequal'], 'dissimilarity': ['similarity'], 'adscript': ['superior'], 'inferior': ['superior'], 'subscript':
['superior'], 'inferiority': ['superiority'], 'insignificance': ['significance'], 'daughter': ['son'], 'dissimilar':
['similar'], 'yield': ['stand'], 'fewer': ['more'], 'unknowable': ['knowable'], 'imperceptible': ['perceptible'],
'inaccurately': ['accurately'], 'unnecessarily': ['necessarily'], 'unconnectedness': ['connexion'], 'nonreciprocal':
['reciprocal'], 'unoriginal': ['original'], 'inessential': ['necessary', 'essential', 'necessity'], 'unnecessary':
['necessary'], 'inadequately': ['adequately'], 'disconnect': ['connect'], 'unplug': ['connect'], 'rear': ['head'],
'tail': ['head'], 'worn': ['new'], 'subtract': ['add'], 'unintelligibly': ['clearly'], 'quadrupedal': ['biped'],
'arrive': ['leave'], 'deposit': ['withdraw'], 'engage': ['withdraw'], 'continue': ['cease'], 'imprecisely':
['exactly'], 'uneasy': ['easy'], 'misconception': ['construct'], 'unusually': ['usually', 'ordinarily'],
'manageable': ['difficult'], 'end': ['begin'], 'unevenly': ['equally', 'evenly'], 'round': ['square'], 'crooked':
['square', 'straight'], 'coldness': ['heat'], 'anestrus': ['heat'], 'unpleasantness': ['sweetness'], 'safe':
['out'], 'impossibility': ['possibility'], 'inside': ['outside'], 'indoor': ['outside'], 'incomplete': ['complete'],
'lower': ['raise'], 'demote': ['raise', 'advance'], 'level': ['raise'], 'penalty': ['advantage'], 'disadvantage':
['advantage'], 'immoderate': ['moderate'], 'physical': ['mental'], 'lodge': ['dislodge'], 'slowly': ['quickly',
'speedily'], 'synonym': ['opposite'], 'alternate': ['opposite'], 'indispose': ['dispose', 'incline'],
'stay_in_place': ['go', 'move'], 'be_born': ['go'], 'no-go': ['go'], 'stop': ['go'], 'impermanent': ['permanent'],
'antitype': ['type'], 'nonvolatile': ['volatile'], 'nonspecific': ['specific'], 'validate': ['avoid'], 'confront':
['avoid'], 'victory': ['defeat'], 'distinctness': ['softness'], 'volume': ['softness'], 'fitness': ['softness'],
'disable': ['enable'], 'surrender': ['withstand'], 'absence': ['presence'], 'sour': ['sweet'], 'dry': ['sweet'],
'salty': ['sweet'], 'unashamed': ['ashamed'], 'unafraid': ['afraid'], 'improbable': ['probable'], 'honor':
['shame'], 'supernatural': ['natural'], 'flat': ['natural'], 'sharp': ['natural'], 'artificial': ['natural'],
'unnatural': ['natural'], 'effective': ['ineffective'], 'unconstitutional': ['constitutional'], 'fearlessness':
['fear'], 'unconstitutionally': ['constitutionally'], 'sanity': ['insanity'], 'normal': ['abnormal'], 'reassure':
['vex'], 'keep': ['lose'], 'win': ['lose'], 'gain': ['lose', 'red'], 'break_even': ['lose'], 'profit': ['lose'],
'unreasonably': ['somewhat', 'reasonably'], 'indirectness': ['straightness'], 'crookedness': ['straightness'],
'curliness': ['straightness'], 'curly': ['straight'], 'indirectly': ['straight', 'directly'], 'curved':
['straight'], 'coiled': ['straight'], 'straight_line': ['curve'], 'distribution': ['density'], 'roughen':
['smooth'], 'staccato': ['smooth'], 'cut': ['rough'], 'substantive': ['adjective'], 'inconsideration':
['consideration'], 'unbox': ['boxing'], 'unerect': ['upright'], 'rarely': ['often'], 'infrequently': ['often'],
'equitable': ['unjust'], 'fair': ['unjust'], 'sad': ['glad'], 'unintelligible': ['intelligible'], 'terminal':
['intermediate'], 'enjoy': ['suffer'], 'be_well': ['suffer'], 'toothed': ['toothless'], 'sighted': ['blind'],
'death': ['birth'], 'negate': ['affirm'], 'prosecution': ['denial'], 'unreceptive': ['receptive'], 'nonmember':
['member'], 'retreat': ['advance', 'progress'], 'regress': ['advance', 'progress'], 'worsen': ['improve'],
'stand_still': ['move'], 'decline': ['improvement'], 'existent': ['nonexistent'], 'unbend': ['turn'], 'impossibly':
['possibly'], 'potential': ['actual'], 'secondarily': ['primarily'], 'young': ['old'], 'obverse': ['reverse'],
'anterior': ['posterior'], 'dishonorable': ['honourable'], 'posteriority': ['priority'], 'dishonor': ['honour'],
'hate': ['love'], 'multiply': ['divide'], 'competent': ['unqualified'], 'qualified': ['unqualified'], 'decrease':
['increase'], 'augmentation': ['diminution'], 'motionlessness': ['motion'], 'subtraction': ['addition'],
'disassemble': ['piece'], 'undress': ['apparel'], 'derive':['differ'], 'open_chain': ['ring'], 'discontented': ['content'], 'discontent':
['content'], 'waste': ['husband'], 'close': ['remote'], 'dead': ['live'], 'recorded': ['live'], 'free':['wild']}}

# additional lemmata as lists!
additional_antonyms = {'fight': ['embrace', 'wander'],
                       'wander': ['fight'],
                       'jacket': ['jersey'],
                       'jersey': ['jacket'],
                       'black': ['blue', 'white'],
                       'blue': ['black', 'white'],
                       'white': ['pink'],
                       'red': ['blue'],
                       'sit': ['run', 'stand'],
                       'run': ['sleep', 'walk'],
                       'play': ['sleep'],
                       'chase': ['sleep'],
                       'jump': ['sleep', 'swim', 'walk'],
                       'swim': ['drown'],
                       'read': ['write'],
                       'write': ['read'],
                       'lay': ['swing'],
                       'talk': ['sleep'],
                       'look': ['sleep', 'asleep'],
                       'inside': ['outside'],
                       'outside': ['room'],
                       'eat': ['throw'],
                       'asleep': ['walk'],
                       'indoor': ['outdoor'],
                       'gentleman': ['pirate'],
                       'girl': ['man', 'boy'],
                       'asian': ['italian'],
                       'old': ['little', 'young'],
                       ('play', 'with'): [('reach', 'up')],
                       ('reach', 'up'): [('play', 'with')],
                       ('do', '*', 'cartwheel'): [('fix', '*', 'home')],
                       ('perform', 'surgery'): [('have', 'lunch')],
                       ('play', 'golf'): [('trade', 'pokemon')],
                       ('sing', '*', 'microphone'): [('play', '*', 'saxophone'), ('play', '*', 'trumpet')],
                       ('work', 'on', '*', 'piece'): [('make', '*', 'pool')],
                       ('hold', '*', 'toy'): [('light', '*', 'thing')],
                       ('be', 'on', '*', 'boat'): [('ride', '*', 'bycicle')],
                       ('prepare', '*', 'balloon'): [('prepare', '*', 'car')],
                       ('be', 'catche', '*', 'fish'): [('compete', '*', 'race')],
                       ('be', '*', 'amusement', 'ride'): [('car', '*', 'broke')],
                       ('be', 'sick'): [('be', 'sing')],
                       ('slide', 'down'): [('pierce', '*', 'knife')],
                       ('play', '*', 'drums'): [('play', '*', 'trumpet')],
                       'wear': ['loose'],
                       ('cut', 'ribbon'): [('draw')],
                       ('sit', 'on', '*', 'step'): [('sit', '*', 'desk')],
                       'exercise': ['eat'],
                       'stand': [('play', '*', 'soccer')],
                       ('coordinate'):[('subordinate')],
                       ('some'):[('generally')],
                       ('have', '*', 'in', 'common'): [('differ', 'in'), 'differ', 'derive'],
                       ('produce'):[('consume')],
                       ('lowlevel'):[('highlevel')],
                       ('own'):[('other')]
                       }

import dict_tools

antonym_dict['lemma_'].update(additional_antonyms)
antonym_dict['lemma_'] = \
    dict_tools.balance_complex_tuple_dict(antonym_dict['lemma_'])



if __name__ == "__main__":
    doctest.testmod()
