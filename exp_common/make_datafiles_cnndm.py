import sys
import os
import hashlib
import struct
import subprocess
import collections
from tqdm import tqdm



dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

# baogs: follow the preprocessing in Contextualized Rewriting
def preprocess(line):
  if len(line) == 0:
    return []

  # restore brackets
  escapes = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}'}
  for key in escapes:
    line = line.replace(key, escapes[key])

  # break paragraph into sentences
  sent_ends = [".", "!", "?"]
  quot_starts = ["``", "`", "(", "[", "{"]
  quot_ends = ["''", "'", ")", "]", "}"]
  sent_end = False
  quot_in = 0
  sents = [[]]
  for w in line.split():
    if w in quot_starts:
      quot_in += 1
    elif w in quot_ends:
      quot_in = quot_in - 1 if quot_in > 0 else 0
    elif w in sent_ends:
      sent_end = True
    sents[-1].append(w)
    if sent_end and quot_in == 0:
      sents.append([])
      sent_end = False

  # `` xxx '' -> " xxx "
  sents = [['"' if w in ["``", "''"] else w for w in sent] for sent in sents if len(sent) > 0]
  sents = [' '.join(sent) for sent in sents]

  # log for checking
  # if len(sents) > 1:
  #   check = ['.', '!', '?']
  #   for c in check:
  #     for sent in sents:
  #       if 0 < sent.find(c) < len(sent) - 1:
  #         print(sent)
  return sents

def get_art_abs(story_file, doc_sep):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]
  lines = sum([preprocess(line) for line in lines], [])

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if len(line) < 5:
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article/abstract into a single string
  article = doc_sep.join(article_lines)
  abstract = doc_sep.join(highlights)
  return article, abstract


def write_to_bin(args, url_file, out_prefix, doc_sep):
  """Reads the .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  with open(out_prefix + '.source', 'wt') as source_file, open(out_prefix + '.target', 'wt') as target_file:
    for idx,s in enumerate(tqdm(story_fnames)):
      # Look in the story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(args.cnn, s)):
        story_file = os.path.join(args.cnn, s)
      elif os.path.isfile(os.path.join(args.dm, s)):
        story_file = os.path.join(args.dm, s)
      else:
        print("Error: Couldn't find story file %s in either story directories %s and %s." % (s, args.cnn, args.dm))
        # Check again if stories directories contain correct number of files
        print("Checking that the stories directories %s and %s contain correct number of files..." % (args.cnn, args.dm))
        check_num_stories(args.cnn, num_expected_cnn_stories)
        check_num_stories(args.dm, num_expected_dm_stories)
        raise Exception("Stories directories %s and %s contain correct number of files but story file %s found in neither." % (args.cnn, args.dm, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file, doc_sep)

      # Write article and abstract to files
      source_file.write(article + '\n')
      target_file.write(abstract + '\n')

  print("Finished writing files")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def main():
  import argparse
  from tqdm import tqdm
  parser = argparse.ArgumentParser()
  parser.add_argument('--urls', default='../shared/cnn_dm/cnn-dailymail-master/url_lists')
  parser.add_argument("--cnn", default='../shared/cnn_dm/cnn_stories_tokenized')
  parser.add_argument("--dm", default='../shared/cnn_dm/dm_stories_tokenized')
  parser.add_argument("--res", default='exp_test/cnndm.tokenized')
  parser.add_argument("--sep", default=' ')
  args, unknown = parser.parse_known_args()

  # Check the stories directories contain the correct number of .story files
  check_num_stories(args.cnn, num_expected_cnn_stories)
  check_num_stories(args.dm, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(args.res):
    os.makedirs(args.res)

  # Read the stories, do a little postprocessing then write to bin files
  write_to_bin(args, "%s/all_test.txt" % args.urls, os.path.join(args.res, "test"), doc_sep=args.sep)
  write_to_bin(args, "%s/all_val.txt" % args.urls, os.path.join(args.res, "valid"), doc_sep=args.sep)
  write_to_bin(args, "%s/all_train.txt" % args.urls, os.path.join(args.res, "train"), doc_sep=args.sep)

if __name__ == '__main__':
  main()