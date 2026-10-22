def test(nums):
    n = len(nums)
    j = 0
    for i in range(n):
        if nums[i] != 0:
            nums[j] = nums[i]
            j+=1
    for i in range(j,n):
        nums[i] = 0
    return nums

print(test([1,0,2,0,3,0,12]))

def three(nums):
    n = len(nums)
    nums.sort()
    ans =[]
    for i in range(n):
        if i>0 and nums[i] ==nums[i-1]:
            continue
        j = i+1
        k = n-1
        while j<k:
            s =nums[i]+nums[j] +nums[k]
            if s<0:
                j+=1
            elif s>0:
                k-=1
            else:
                ans.append([nums[i],nums[j],nums[k]])
                j+=1
                k-=1
                while j<k and nums[j] == nums[j-1]:
                    j+=1
                while j<k and nums[k] == nums[k+1]:
                    k-=1
    return ans

print(three([-1,0,1,3,-1,2]))


def water(nums):
    n = len(nums)
    l=0
    r=n-1
    s=0
    ans=0
    while l<r:
        s=(r-l)*min(nums[l],nums[r])
        ans=max(s,ans)
        if nums[l] < nums[r]:
            l+=1
        else:
            r-=1
    return ans

print(water([4,3,2,1,2,3,4]))

def rain(nums):
    n = len(nums)
    l=0
    r=n-1
    lmax=0
    rmax=0
    ans = 0
    while l<r:
        lmax=max(lmax,nums[l])
        rmax=max(rmax,nums[r])
        if lmax <rmax:
            ans+=lmax-nums[l]
            l+=1
        else:
            ans+=rmax-nums[r]
            r-=1
    return ans
print(rain([4,3,2,1,2,3,4]))


def test_min(nums,k):
    n = len(nums)
    j = 0
    ans = n+1
    s = 0
    for i,num in enumerate(nums):
        s+=num 
        while s>=k:
            ans=min(ans,i-j+1)
            s-=nums[j]
            j+=1
    return ans

print(test_min([1,2,3,4,5],6))


def test_cj(nums,k):
    j = 0
    s=1
    ans =0
    for i,num in enumerate(nums):
        s*=num
        while s>=k:
            s/=nums[j]
            j+=1
        ans+=i-j+1
    return ans
print(test_cj([1,2,3,4,5],6))
