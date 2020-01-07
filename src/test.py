def is_merge(s, part1, part2):
    s_idx = 0
    one_idx = 0
    two_idx = 0
    notfound = 0

    for idx in range(len(s)):
        print(idx)
        if len(part1)-1 >= one_idx:
            if s[idx] == part1[one_idx]:
                print(part1[one_idx])
                one_idx += 1
                notfound = 0
            else:
                notfound += 1
        if len(part2)-1 >= two_idx:
            if s[idx] == part2[two_idx]:
                print(part2[two_idx])
                two_idx += 1
                notfound = 0
            else:
                notfound += 1
        if notfound == 2:
            return False
    return True
            


if __name__ == "__main__":
    print(is_merge('codewars', 'code', 'wars'))
    print(is_merge('codewars', 'cdw', 'oears'))
    print(is_merge('codewars', 'cod', 'wars'))