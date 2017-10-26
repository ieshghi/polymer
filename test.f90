program test
    use lattice 
    implicit none
    integer::n,zer,one,two,thr,i
    integer,dimension(:),allocatable::arr
    n = 10

    arr = genpoly(n)
    zer = 0
    one = 0
    two = 0
    thr = 0
    do i=1,n
        if(arr(i)==0)then
            zer = zer+1
        elseif(arr(i)==1)then
            one = one+1
        elseif(arr(i)==2)then
            two = two+1
        elseif(arr(i)==3)then
            thr = thr+1
        endif
    enddo
    write(*,*) zer,one,two,thr,n
endprogram test
