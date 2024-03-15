import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import DataTableComponent from '../../Component/DataTableComponent'
import Constant from '../../Constants/validator';
import TotalRow from '../../Component/TotalRow';
import color from '../../Constants/color';
import '../../App.scss';
import './RouteProfitabilityConsolidate.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import cookieStorage from '../../Constants/cookie-storage';
import BrowserToProps from 'react-browser-to-props';
import Spinners from "../../spinneranimation";
import { result } from 'lodash';
import DownloadCSV from '../.././Component/DownloadCSV'

const apiServices = new APIServices();

class RouteProfitabilityConsolidate extends Component {
  constructor(props) {
    super(props);
    this.userDetails = JSON.parse(cookieStorage.getCookie('userDetails'))
    this.state = {
      bcData: [],
      startDate: '',
      endDate: '',
      gettingDay: null,
      gettingMonth: null,
      gettingYear: null,
      routeGroup: '',
      regionId: '*',
      countryId: '*',
      routeId: '*',
      flight: '*',
      headerData: [],
      headerColumns: [],
      routeMonthDetails: [],
      routeMonthColumns: [],
      monthTotalData: [],
      revenue: [],
      revenueColumns: [],
      revenueTotal: [],
      vc: [],
      vcColumns: [],
      vcTotal: [],
      doc: [],
      docColumns: [],
      docTotal: [],
      ContributionCost: [],
      totalCost: [],
      vcContribution: [],
      docContribution: [],
      noneRoute: [],
      surplusDeficit: [],
      totalOtherFixCost: [],
      NIAT: [],
      routeCommercial: [],
      routeCommercialTotal: [],
      loadingNonCommercial: false,
      loadingCommercial: false,
      routeMonthRowClassRule: {
        'highlight-row': 'data.highlightMe',
      },
    }
    Constant.sendEvent('1', 'viewed Route Profitability Consolidate Page', '/routeProfitabilityConsolidate', 'Route Profitability Consolidate');
  }

  componentDidMount() {
    var self = this;

    const header = apiServices.getRPCHeader();
    self.setState({ headerData: header[0].rowData, headerColumns: header[0].columnName })

    self.getFiltersValue();
  }

  getFiltersValue = () => {
    let routeGroup = window.localStorage.getItem('RouteGroupSelected')
    let RegionSelected = window.localStorage.getItem('RouteRegionSelected')
    let CountrySelected = window.localStorage.getItem('RouteCountrySelected')
    let RouteSelected = window.localStorage.getItem('RouteSelected')
    let dateRange = JSON.parse(window.localStorage.getItem('RRDateRangeValue'))
    let endDateArray = dateRange[0].endDate.split('-')
    let RRDay = window.localStorage.getItem('RRDay')
    let FlightSelected = window.localStorage.getItem('RPFlightSelected')

    if (routeGroup === null || routeGroup === '' || routeGroup === 'Null') {
      if (Object.keys(this.userDetails.route_access).length > 0) {
        routeGroup = this.userDetails.route_access['selectedRouteGroup']
      } else {
        routeGroup = ['Network']
      }
    } else {
      routeGroup = JSON.parse(routeGroup)
    }
    this.getLegends(dateRange);
    this.setState({
      routeGroup: routeGroup.join("','"),
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
      routeId: RouteSelected === null || RouteSelected === 'Null' || RouteSelected === '' ? '*' : JSON.parse(RouteSelected),
      flight: FlightSelected === null || FlightSelected === 'Null' || FlightSelected === '' ? '*' : `'${FlightSelected}'`,
      gettingYear: endDateArray[0],
      gettingMonth: window.monthNumToName(endDateArray[1]),
      gettingDay: RRDay === 'false' ? null : endDateArray[2],
    }, () => this.getInitialData())
  }

  getInitialData() {
    let self = this;
    let { routeGroup, gettingDay, gettingMonth, gettingYear, regionId, countryId, routeId, flight } = this.state;

    self.setState({ loadingNonCommercial: true, loadingCommercial: true })

    apiServices.getRPNonCommercialInfo(gettingYear, gettingMonth, gettingDay, routeGroup, regionId, countryId, routeId, flight, 'Null').then((result) => {
      this.setState({ loadingNonCommercial: false })
      if (result) {
        self.setState({
          revenue: result[0].revenue,
          revenueTotal: result[0].revenueTotal,
          vc: result[0].vc,
          vcTotal: result[0].vcTotal,
          vcContribution: result[0].vcContribution,
          doc: result[0].doc,
          docTotal: result[0].docTotal,
          docContribution: result[0].docContribution,
          totalOtherFixCost: result[0].totalOtherFixCost,
          totalCost: result[0].totalCost,
          surplusDeficit: result[0].surplusDeficit,
          noneRoute: result[0].noneRoute,
          NIAT: result[0].NIAT,
        })
      }
    })

    apiServices.getRPCommercialInfo(gettingYear, gettingMonth, gettingDay, routeGroup, regionId, countryId, routeId, flight, 'Null').then((result) => {
      this.setState({ loadingCommercial: false })
      if (result) {
        self.setState({ routeCommercial: result[0].rowData, routeCommercialTotal: result[0].total })
      }
    })
  }

  getLegends(dateRange) {
    let bcData = window.localStorage.getItem('RRBCData')
    bcData = bcData ? JSON.parse(bcData) : []
    let startDate = dateRange[0].startDate.split('-')
    let endDate = dateRange[0].endDate.split('-')
    startDate = new Date(startDate[0], startDate[1] - 1, startDate[2])
    endDate = new Date(endDate[0], endDate[1] - 1, endDate[2])
    this.setState({ bcData, startDate: Constant.getDateFormat(startDate), endDate: Constant.getDateFormat(endDate) })
  }

  render() {
    const downloadURL = localStorage.getItem('RouteProfitabilityNCDownloadURL')
    const downloadURLC = localStorage.getItem('RouteProfitabilityCDownloadURL')
    return (
      <div className="route-profitabilty-consolidate-page">
        <TopMenuBar dashboardPath={'/routeProfitability'} {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign">
              <div className="col-md-7 col-sm-7 col-xs-12 toggle1">
                <h2>Route Profitability Consolidate</h2>
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      <p>{`${this.state.routeGroup} `}</p>
                      {this.state.bcData.map((item) =>
                        <div style={{ cursor: item.disable ? 'not-allowed' : 'pointer' }}>
                          <li id={item.val} title={`${item.title} : ${item.val}`}>
                            &nbsp;{` > ${item.val}`}
                          </li>
                        </div>
                      )}
                    </ol>
                  </nav>
                </section>
              </div>

              <div className="col-md-5 col-sm-5 col-xs-12 toggle2">
                <span className='Rightheader' >Date Range : {`${this.state.startDate} - ${this.state.endDate}`}</span>
                <DownloadCSV url={downloadURLC} name={`Commercial`} path={`/routeProfitabilityConsolidate`} page={`RouteProfitability`} />
                <DownloadCSV url={downloadURL} name={`Non-Commercial`} path={`/routeProfitabilityConsolidate`} page={`RouteProfitability`} />
              </div>

            </div>

          </div>
        </div>
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div className="x_panel" >
              <div className="x_content">
                <div className="tables-main">
                  <div className="tableHeader">
                    <DataTableComponent
                      rowData={this.state.headerData}
                      columnDefs={this.state.headerColumns}
                      autoHeight={'autoHeight'}
                      route={true}
                    />
                  </div>
                  <div className="tableContent" >
                    <h6>Revenue</h6>
                    <DataTableComponent
                      rowData={this.state.revenue}
                      columnDefs={this.state.headerColumns}
                      autoHeight={'autoHeight'}
                      hideHeader={'0'}
                      rowClassRules={this.state.routeMonthRowClassRule}
                      route={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.revenueTotal}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <h6>Variable Cost</h6>
                    <DataTableComponent
                      rowData={this.state.vc}
                      columnDefs={this.state.headerColumns}
                      hideHeader={'0'}
                      autoHeight={'autoHeight'}
                      rowClassRules={this.state.routeMonthRowClassRule}
                      route={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.vcTotal}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.vcContribution}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                      changeBgColor={true}
                    />
                    <h6>Direct Fixed Cost</h6>
                    <DataTableComponent
                      rowData={this.state.doc}
                      columnDefs={this.state.headerColumns}
                      hideHeader={'0'}
                      autoHeight={'autoHeight'}
                      rowClassRules={this.state.routeMonthRowClassRule}
                      route={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.docTotal}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.docContribution}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                      changeBgColor={true}
                    />
                    <TotalRow
                      rowData={this.state.totalOtherFixCost}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.totalCost}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                      changeBgColor={true}
                    />
                    <TotalRow
                      rowData={this.state.surplusDeficit}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <TotalRow
                      rowData={this.state.noneRoute}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                      changeBgColor={true}
                    />
                    <TotalRow
                      rowData={this.state.NIAT}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingNonCommercial}
                    />
                    <h6>Commercial Info</h6>
                    <DataTableComponent
                      rowData={this.state.routeCommercial}
                      columnDefs={this.state.headerColumns}
                      hideHeader={'0'}
                      height={'60vh'}
                      autoHeight={'autoHeight'}
                      rowClassRules={this.state.routeMonthRowClassRule}
                      route={true}
                      loading={this.state.loadingCommercial}
                    />
                    {/* <TotalRow
                      rowData={this.state.routeCommercialTotal}
                      columnDefs={this.state.headerColumns}
                      responsive={true}
                      loading={this.state.loadingCommercial}
                    /> */}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
}
const NewComponent = BrowserToProps(RouteProfitabilityConsolidate);


export default NewComponent;