# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

'''
ANSI font and color codes.
'''
import sys

ansi_esc_ = chr(0x1B) + '['

#-------------------------------------------------------------------------------
class Font( object ):
    '''
    ANSI font and color codes.

    The Font class implements all settings, and is accessible by a global object
    `font`. Member functions that take a string `msg`, e.g., `font.bold( msg )`,
    return a string of an ANSI code + msg + an ANSI code to restore the state.
    Member functions that don't take a string just return an ANSI code, usually
    to restore the state, e.g., `font.not_italic()`. Example use:

        from ansicodes import font
        font.set_enabled( True )  # enabled by default
        print( font.bold( font.red( 'red' ) + ' bold' ) + ' normal.' )

    Styles and one color can be nested:

        print( 'normal'
               + font.bold( 'bold '
                            + font.italic( 'bold-italic '
                                           + fg.red( 'red-bold-italic' )
                                           + ' bold-italic' )
                            + ' bold' )
               + ' normal' )

    Multiple colors cannot be nested:

        print( fg.red( 'red '
                       + fg.blue( 'blue' )
                       + ' *black' )
               + ' black' )

    where one might expect the ' *black' text to be red, but that color is lost.
    '''
    #--------------------
    def __init__( self ):
        self.enabled_ = True

    #--------------------
    def set_enabled( self, val ):
        '''
        Enable or disable ANSI codes.
        Enable if val == True, 'y', 'yes', or 'always',
        or if val = 'auto' and output is to a console (TTY).
        '''
        self.enabled_ = (val in (True, 'y', 'yes', 'always')
                         or (val == 'auto' and sys.stdout.isatty()))
    # end

    #--------------------
    def code( self, val ):
        '''
        If ANSI codes are enabled, returns "ESC[" + val + "m";
        otherwise returns empty string.
        '''
        if (self.enabled_):
            return ansi_esc_ + val + 'm'
        else:
            return ''
    # end

    #-------------------- font styles
    def      reset( self, msg ): return self.code('0')
    def       bold( self, msg ): return self.code('1')  + msg + self.normal()
    def      faint( self, msg ): return self.code('2')  + msg + self.normal()
    def     italic( self, msg ): return self.code('3')  + msg + self.not_italic()
    def  underline( self, msg ): return self.code('4')  + msg + self.not_underline()
    def      blink( self, msg ): return self.code('5')  + msg + self.steady()
    def blink_fast( self, msg ): return self.code('6')  + msg + self.steady()
    def   negative( self, msg ): return self.code('7')  + msg + self.positive()
    def    conceal( self, msg ): return self.code('8')  + msg + self.reveal()
    def     strike( self, msg ): return self.code('9')  + msg + self.not_strike()
    def    fraktur( self, msg ): return self.code('20') + msg + self.not_italic()

    # 21 is either not-bold or double underline, an unfortunate ambiguity.
    def      not_bold( self ): return self.code('21')
    def        normal( self ): return self.code('22')  # not bold, not faint
    def    not_italic( self ): return self.code('23')
    def   not_fraktur( self ): return self.code('23')
    def not_underline( self ): return self.code('24')
    def        steady( self ): return self.code('25')
    def      positive( self ): return self.code('27')
    def        reveal( self ): return self.code('28')
    def    not_strike( self ): return self.code('29')

    #-------------------- fonts (rarely supported)
    def default_font( self ): return self.code('10')
    def font1( self, msg ): return self.code('11') + msg + self.default_font()
    def font2( self, msg ): return self.code('12') + msg + self.default_font()
    def font3( self, msg ): return self.code('13') + msg + self.default_font()
    def font4( self, msg ): return self.code('14') + msg + self.default_font()
    def font5( self, msg ): return self.code('15') + msg + self.default_font()
    def font6( self, msg ): return self.code('16') + msg + self.default_font()
    def font7( self, msg ): return self.code('17') + msg + self.default_font()
    def font8( self, msg ): return self.code('18') + msg + self.default_font()
    def font9( self, msg ): return self.code('19') + msg + self.default_font()

    #-------------------- frames (rarely supported)
    def    framed( self, msg ): return self.code('51') + msg + self.not_framed()
    def encircled( self, msg ): return self.code('52') + msg + self.not_encircled()
    def overlined( self, msg ): return self.code('53') + msg + self.not_overlined()
    def     not_framed( self ): return self.code('54')
    def  not_encircled( self ): return self.code('54')
    def  not_overlined( self ): return self.code('55')

    #-------------------- foreground colors
    def   black( self, msg ): return self.code('30') + msg + self.default_color()
    def     red( self, msg ): return self.code('31') + msg + self.default_color()
    def   green( self, msg ): return self.code('32') + msg + self.default_color()
    def  yellow( self, msg ): return self.code('33') + msg + self.default_color()
    def    blue( self, msg ): return self.code('34') + msg + self.default_color()
    def magenta( self, msg ): return self.code('35') + msg + self.default_color()
    def    cyan( self, msg ): return self.code('36') + msg + self.default_color()
    def    gray( self, msg ): return self.code('37') + msg + self.default_color()
    def default_color( self ): return self.code('39')

    def color( self, r, g, b, msg ):
        '''
        Returns string to display msg using a 24-bit RGB foreground color.
        Supported on Xterm, Konsole, Gnome, libvte, etc.
        May produce weird results on others like macOS Terminal.
        '''
        return self.code('38;2;%d;%d;%d' % (r, g, b)) + msg + self.default_color()

    #-------------------- background colors
    def   black_bg( self, msg ): return self.code('40') + msg + self.default_bgcolor()
    def     red_bg( self, msg ): return self.code('41') + msg + self.default_bgcolor()
    def   green_bg( self, msg ): return self.code('42') + msg + self.default_bgcolor()
    def  yellow_bg( self, msg ): return self.code('43') + msg + self.default_bgcolor()
    def    blue_bg( self, msg ): return self.code('44') + msg + self.default_bgcolor()
    def magenta_bg( self, msg ): return self.code('45') + msg + self.default_bgcolor()
    def    cyan_bg( self, msg ): return self.code('46') + msg + self.default_bgcolor()
    def    gray_bg( self, msg ): return self.code('47') + msg + self.default_bgcolor()
    def default_bgcolor( self ): return self.code('49')

    def bgcolor( self, r, g, b, msg ):
        '''
        Returns string to display msg using a 24-bit RGB background color.
        '''
        return self.code('48;2;%d;%d;%d' % (r, g, b)) + msg + self.default_bgcolor()
# end

#-------------------------------------------------------------------------------
# Global object to access the Font class.

font = Font()

#-------------------------------------------------------------------------------
def test():
    print( font.bold( 'Styles' ) )
    print( 'bold:      ', font.bold      ( ' text ' ), 'post' )
    print( 'faint:     ', font.faint     ( ' text ' ), 'post' )
    print( 'italic:    ', font.italic    ( ' text ' ), 'post' )
    print( 'underline: ', font.underline ( ' text ' ), 'post' )
    print( 'blink:     ', font.blink     ( ' text ' ), 'post' )
    print( 'blink_fast:', font.blink_fast( ' text ' ), 'post *' )
    print( 'negative:  ', font.negative  ( ' text ' ), 'post' )
    print( 'conceal:   ', font.conceal   ( ' text ' ), 'post' )
    print( 'strike:    ', font.strike    ( ' text ' ), 'post' )
    print( 'fraktur:   ', font.fraktur   ( ' text ' ), 'post *' )
    print( 'framed:    ', font.framed    ( ' text ' ), 'post *' )
    print( 'encircled: ', font.encircled ( ' text ' ), 'post *' )
    print( 'overlined: ', font.overlined ( ' text ' ), 'post *' )

    print( font.bold( '\nFonts (* rarely supported)' ) )
    print( 'font1:', font.font1( ' text ' ), 'post *' )
    print( 'font2:', font.font2( ' text ' ), 'post *' )
    print( 'font3:', font.font3( ' text ' ), 'post *' )
    print( 'font4:', font.font4( ' text ' ), 'post *' )
    print( 'font5:', font.font5( ' text ' ), 'post *' )
    print( 'font6:', font.font6( ' text ' ), 'post *' )
    print( 'font7:', font.font7( ' text ' ), 'post *' )
    print( 'font8:', font.font8( ' text ' ), 'post *' )
    print( 'font9:', font.font9( ' text ' ), 'post *' )

    print( font.bold( '\nForeground colors' ) )
    print( 'default:', font.default_color() + ' text ', 'post' )
    print( 'black:  ', font.black  ( ' text ' ), 'post' )
    print( 'red:    ', font.red    ( ' text ' ), 'post' )
    print( 'green:  ', font.green  ( ' text ' ), 'post' )
    print( 'yellow: ', font.yellow ( ' text ' ), 'post' )
    print( 'blue:   ', font.blue   ( ' text ' ), 'post' )
    print( 'magenta:', font.magenta( ' text ' ), 'post' )
    print( 'cyan:   ', font.cyan   ( ' text ' ), 'post' )
    print( 'gray:   ', font.gray   ( ' text ' ), 'post' ) # aka, "white"
    print( font.bold( 'RGB colors' ) )
    print( 'red:    ', font.color( 255,   0,   0, ' text ' ), 'post' )
    print( 'purple: ', font.color( 148,  33, 147, ' text ' ), 'post' )
    print( 'orange: ', font.color( 255, 147,   0, ' text ' ), 'post' )

    print( font.bold( '\nForeground colors + bold ("bright")' ) )
    print( 'default:', font.bold( font.default_color() + ' text ' ), 'post' )
    print( 'black:  ', font.bold( font.black  ( ' text ' ) ), 'post' )
    print( 'red:    ', font.bold( font.red    ( ' text ' ) ), 'post' )
    print( 'green:  ', font.bold( font.green  ( ' text ' ) ), 'post' )
    print( 'yellow: ', font.bold( font.yellow ( ' text ' ) ), 'post' )
    print( 'blue:   ', font.bold( font.blue   ( ' text ' ) ), 'post' )
    print( 'magenta:', font.bold( font.magenta( ' text ' ) ), 'post' )
    print( 'cyan:   ', font.bold( font.cyan   ( ' text ' ) ), 'post' )
    print( 'gray:   ', font.bold( font.gray   ( ' text ' ) ), 'post' )
    print( font.bold( 'RGB colors' ) )
    print( 'red:    ', font.bold( font.color( 255,   0,   0, ' text ' ) ), 'post' )
    print( 'purple: ', font.bold( font.color( 148,  33, 147, ' text ' ) ), 'post' )
    print( 'orange: ', font.bold( font.color( 255, 147,   0, ' text ' ) ), 'post' )

    print( font.bold( '\nBackground colors' ) )
    print( 'bg default:', font.default_bgcolor() + ' text ', 'post' )
    print( 'bg black:  ', font.black_bg  ( ' text ' ), 'post' )
    print( 'bg red:    ', font.red_bg    ( ' text ' ), 'post' )
    print( 'bg green:  ', font.green_bg  ( ' text ' ), 'post' )
    print( 'bg yellow: ', font.yellow_bg ( ' text ' ), 'post' )
    print( 'bg blue:   ', font.blue_bg   ( ' text ' ), 'post' )
    print( 'bg magenta:', font.magenta_bg( ' text ' ), 'post' )
    print( 'bg cyan:   ', font.cyan_bg   ( ' text ' ), 'post' )
    print( 'bg gray:   ', font.gray_bg   ( ' text ' ), 'post' )
    print( font.bold( 'bg RGB colors' ) )
    print( 'bg red:    ', font.bgcolor( 255,   0,   0, ' text ' ), 'post' )
    print( 'bg purple: ', font.bgcolor( 148,  33, 147, ' text ' ), 'post' )
    print( 'bg orange: ', font.bgcolor( 255, 147,   0, ' text ' ), 'post' )

    print( font.bold( '\nBackground colors + bold ("bright")' ) )
    print( 'bg black:  ', font.bold( font.black_bg  ( ' text ' ) ), 'post' )
    print( 'bg red:    ', font.bold( font.red_bg    ( ' text ' ) ), 'post' )
    print( 'bg green:  ', font.bold( font.green_bg  ( ' text ' ) ), 'post' )
    print( 'bg yellow: ', font.bold( font.yellow_bg ( ' text ' ) ), 'post' )
    print( 'bg blue:   ', font.bold( font.blue_bg   ( ' text ' ) ), 'post' )
    print( 'bg magenta:', font.bold( font.magenta_bg( ' text ' ) ), 'post' )
    print( 'bg cyan:   ', font.bold( font.cyan_bg   ( ' text ' ) ), 'post' )
    print( 'bg gray:   ', font.bold( font.gray_bg   ( ' text ' ) ), 'post' )
    print( font.bold( 'bg RGB colors' ) )
    print( 'bg red:    ', font.bold( font.bgcolor( 255,   0,   0, ' text ' ) ), 'post' )
    print( 'bg purple: ', font.bold( font.bgcolor( 148,  33, 147, ' text ' ) ), 'post' )
    print( 'bg orange: ', font.bold( font.bgcolor( 255, 147,   0, ' text ' ) ), 'post' )

    print( font.bold( '\nCombinations' ) )
    # bold + italic, bold + underline
    print( 'pre '
           + font.bold( 'bold ' + font.italic( 'bold-italic' ) + ' bold' )
           + ' normal' )
    print( 'pre '
           + font.bold( 'bold ' + font.underline( 'bold-underline' ) + ' bold' )
           + ' normal' )

    # italic + bold, italic + underline
    print( 'pre '
           + font.italic( 'italic ' + font.bold( 'italic-bold' ) + ' italic' )
           + ' normal' )
    print( 'pre '
           + font.italic( 'italic ' + font.underline( 'italic-underline' )
               + ' italic' )
           + ' normal' )

    # underline + bold, underline + italic
    print( 'pre '
           + font.underline( 'underline ' + font.bold( 'underline-bold' )
               + ' underline' )
           + ' normal' )
    print( 'pre '
           + font.underline( 'underline ' + font.italic( 'underline-italic' )
               + ' underline' )
           + ' normal' )

    # bold + italic + underline
    print( 'pre '
           + font.bold( 'bold ' + font.underline( 'bold-underline '
               + font.italic( 'bold-italic-underline' ) + ' bold-underline' )
               + ' bold' )
           + ' normal' )

    # bold + fg color, bold + bg color
    print( 'pre '
           + font.bold( 'bold ' + font.red( 'bold-red' ) + ' bold' )
           + ' normal' )
    print( 'pre '
           + font.bold( 'bold ' + font.red_bg( 'bold/red' ) + ' bold' )
           + ' normal' )

    # colors: fg + bg, bg + fg, negative bg + fg
    print( 'pre '
           + font.red( 'red ' + font.yellow_bg( 'red/yellow' ) + ' red' )
           + ' normal' )
    print( 'pre '
           + font.red_bg( 'black/red ' + font.yellow( 'yellow/red' ) + ' black/red' )
           + ' normal' )
    print( 'pre '
           + font.negative( 'negative '
           + font.red_bg( 'black/red ' + font.yellow( 'yellow/red' ) + ' black/red' )
           + ' negative' ) + ' normal' )
# end
