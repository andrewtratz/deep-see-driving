Question 11

def CheckInValidChar(s):
    if s == "" or s is None:
        return False
    else:
        # check if first char, last char
        if s[0] !="H" or s[-1 ]=="H":
            return False
        try:
            if s[1] == '0':
                return False
            if s[2] == 'A':
                return False
        except:
            return True
    return True


print(CheckInValidChar('H1A'))